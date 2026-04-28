#!/usr/bin/env python3
"""
Batch Manager: Download completed chunks, submit pending chunks, merge results.
Handles the multi-chunk batch workflow for LLM classification.
"""

import os
import sys
import json
import time
import pandas as pd

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PROJECT_ROOT)

CACHE_DIR = os.path.join(PROJECT_ROOT, 'data', '02_processed', '03_classified', 'llm_cache')
MANIFEST_PATH = os.path.join(CACHE_DIR, 'batch_manifest.json')
MERGED_PATH = os.path.join(PROJECT_ROOT, 'data', '02_processed', '02_merged', 'merged_dataset.csv')


def load_manifest():
    with open(MANIFEST_PATH) as f:
        return json.load(f)


def save_manifest(manifest):
    with open(MANIFEST_PATH, 'w') as f:
        json.dump(manifest, f, indent=2)


def download_chunk(batch_entry, api_key, client):
    """Download results for a single completed chunk."""
    from importlib.util import spec_from_file_location, module_from_spec
    spec = spec_from_file_location(
        "cloud_classification",
        os.path.join(PROJECT_ROOT, 'pipeline', '03_classification', 'cloud_classification.py')
    )
    cc = module_from_spec(spec)
    spec.loader.exec_module(cc)

    batch_id = batch_entry['batch_id']
    chunk_num = batch_entry['chunk']
    n_records = batch_entry['n_records']

    print(f"\n--- Downloading chunk {chunk_num} (batch {batch_id}) ---")

    # Check status
    batch = client.messages.batches.retrieve(batch_id)
    if batch.processing_status != 'ended':
        print(f"  Not ready: {batch.processing_status}")
        return None

    succeeded = batch.request_counts.succeeded
    errored = batch.request_counts.errored
    print(f"  Succeeded: {succeeded:,}, Errored: {errored:,}")

    # Download and parse
    results = [None] * n_records
    error_count = 0
    error_details = []

    for result in client.messages.batches.results(batch_id):
        idx = int(result.custom_id)
        if result.result.type == 'succeeded':
            response_text = result.result.message.content[0].text
            parsed = cc._parse_llm_response(response_text)
        else:
            error_count += 1
            error_type = result.result.type
            if error_count <= 5:
                error_details.append(f"  idx={idx}: {error_type}")
            parsed = {
                'classification': 'error',
                'platform': 'error',
                'confidence': 'low',
                'evidence': f'batch_{error_type}',
            }
        results[idx] = parsed

    # Fill missing
    for i in range(n_records):
        if results[i] is None:
            error_count += 1
            results[i] = {
                'classification': 'error',
                'platform': 'error',
                'confidence': 'low',
                'evidence': 'missing_from_batch',
            }

    # Save per-chunk file
    chunk_path = os.path.join(CACHE_DIR, f'chunk_{chunk_num:02d}_results.csv')
    df = pd.DataFrame(results)
    df.to_csv(chunk_path, index=False)

    print(f"  Saved {n_records:,} results to {chunk_path}")
    print(f"  Errors: {error_count:,}")
    if error_details:
        print(f"  First errors:")
        for e in error_details:
            print(f"    {e}")

    # Classification distribution
    print(f"\n  Classification distribution:")
    for cls, count in df['classification'].value_counts().items():
        print(f"    {cls:25s}: {count:>8,} ({count/n_records*100:.1f}%)")

    return df


def submit_chunk(chunk_entry, api_key, client, merged_df):
    """Submit a pending chunk."""
    from importlib.util import spec_from_file_location, module_from_spec
    spec = spec_from_file_location(
        "cloud_classification",
        os.path.join(PROJECT_ROOT, 'pipeline', '03_classification', 'cloud_classification.py')
    )
    cc = module_from_spec(spec)
    spec.loader.exec_module(cc)

    chunk_num = chunk_entry['chunk']
    start_idx = chunk_entry['start_idx']
    end_idx = chunk_entry['end_idx']
    n_records = chunk_entry['n_records']

    print(f"\n--- Submitting chunk {chunk_num} (rows {start_idx:,}-{end_idx-1:,}, {n_records:,} records) ---")

    chunk_df = merged_df.iloc[start_idx:end_idx].copy()

    # Build requests
    requests = cc._build_batch_requests(chunk_df)
    print(f"  Built {len(requests):,} requests")

    # Submit
    batch = client.messages.batches.create(requests=requests)
    batch_id = batch.id
    print(f"  Submitted: {batch_id}")

    return {
        'batch_id': batch_id,
        'chunk': chunk_num,
        'start_idx': start_idx,
        'end_idx': end_idx,
        'n_records': n_records,
        'status': 'submitted',
    }


def merge_all_chunks(manifest):
    """Merge all chunk CSV files into final results."""
    total_chunks = len(manifest['batches'])
    all_dfs = []

    for i in range(1, total_chunks + 1):
        chunk_path = os.path.join(CACHE_DIR, f'chunk_{i:02d}_results.csv')
        if not os.path.exists(chunk_path):
            print(f"  Missing chunk {i}: {chunk_path}")
            return None
        df = pd.read_csv(chunk_path)
        all_dfs.append(df)
        print(f"  Chunk {i}: {len(df):,} records")

    merged = pd.concat(all_dfs, ignore_index=True)
    final_path = os.path.join(CACHE_DIR, 'llm_results_final.csv')
    merged.to_csv(final_path, index=False)

    # Write progress file for compatibility with pipeline
    progress = {
        'last_completed': len(merged),
        'total': manifest['total_records'],
        'status': 'complete',
        'source': 'multi_batch',
        'errors': int((merged['classification'] == 'error').sum()),
        'last_updated': time.strftime('%Y-%m-%d %H:%M:%S'),
    }
    with open(os.path.join(CACHE_DIR, 'llm_progress.json'), 'w') as f:
        json.dump(progress, f, indent=2)

    print(f"\n  Merged {len(merged):,} total results")
    print(f"  Errors: {progress['errors']:,}")
    print(f"  Saved to: {final_path}")

    # Overall distribution
    print(f"\n  Overall classification distribution:")
    for cls, count in merged['classification'].value_counts().items():
        pct = count / len(merged) * 100
        print(f"    {cls:25s}: {count:>8,} ({pct:.1f}%)")

    return merged


def patch_chunk_with_retry(client, cache_dir, chunk_num):
    """Download retry batch results and patch the specified chunk."""
    retry_path = os.path.join(cache_dir, f'retry_chunk_{chunk_num:02d}.json')
    if not os.path.exists(retry_path):
        return False

    with open(retry_path) as f:
        retry_meta = json.load(f)

    if retry_meta.get('patched'):
        print(f"  Chunk {chunk_num} retry already patched.")
        return True

    batch_id = retry_meta['batch_id']
    batch = client.messages.batches.retrieve(batch_id)

    if batch.processing_status != 'ended':
        print(f"  Retry batch {batch_id}: {batch.processing_status} "
              f"({batch.request_counts.succeeded:,} succeeded, "
              f"{batch.request_counts.processing:,} processing)")
        return False

    print(f"  Downloading retry results ({batch.request_counts.succeeded:,} succeeded, "
          f"{batch.request_counts.errored:,} errored)...")

    # Import parser
    from importlib.util import spec_from_file_location, module_from_spec
    spec = spec_from_file_location(
        "cloud_classification",
        os.path.join(PROJECT_ROOT, 'pipeline', '03_classification', 'cloud_classification.py')
    )
    cc = module_from_spec(spec)
    spec.loader.exec_module(cc)

    # Parse retry results keyed by chunk-local index
    retry_results = {}
    for result in client.messages.batches.results(batch_id):
        local_idx = int(result.custom_id)
        if result.result.type == 'succeeded':
            response_text = result.result.message.content[0].text
            retry_results[local_idx] = cc._parse_llm_response(response_text)
        else:
            retry_results[local_idx] = {
                'classification': 'error',
                'platform': 'error',
                'confidence': 'low',
                'evidence': f'retry_{result.result.type}',
            }

    # Patch the chunk
    chunk_path = os.path.join(cache_dir, f'chunk_{chunk_num:02d}_results.csv')
    chunk_df = pd.read_csv(chunk_path)

    patched = 0
    still_errored = 0
    for local_idx, parsed in retry_results.items():
        if parsed['classification'] != 'error':
            chunk_df.loc[local_idx, 'classification'] = parsed['classification']
            chunk_df.loc[local_idx, 'platform'] = parsed['platform']
            chunk_df.loc[local_idx, 'confidence'] = parsed['confidence']
            chunk_df.loc[local_idx, 'evidence'] = parsed['evidence']
            patched += 1
        else:
            still_errored += 1

    chunk_df.to_csv(chunk_path, index=False)

    retry_meta['patched'] = True
    retry_meta['patched_count'] = patched
    retry_meta['still_errored'] = still_errored
    with open(retry_path, 'w') as f:
        json.dump(retry_meta, f, indent=2)

    print(f"  Patched {patched:,} records, {still_errored:,} still errored")

    # Show updated distribution
    print(f"\n  Updated chunk {chunk_num} distribution:")
    for cls, count in chunk_df['classification'].value_counts().items():
        print(f"    {cls:25s}: {count:>8,} ({count/len(chunk_df)*100:.1f}%)")

    return True


def main():
    api_key = os.environ.get('ANTHROPIC_API_KEY')
    if not api_key:
        print("ERROR: Set ANTHROPIC_API_KEY environment variable")
        sys.exit(1)

    import anthropic
    client = anthropic.Anthropic(api_key=api_key)

    manifest = load_manifest()
    print(f"Total records: {manifest['total_records']:,}")
    print(f"Submitted batches: {len(manifest['batches'])}")
    print(f"Pending chunks: {len(manifest.get('pending_chunks', []))}")

    # Step 1: Check status and download completed batches
    print("\n" + "=" * 60)
    print("STEP 1: Check & download completed batches")
    print("=" * 60)

    for batch_entry in manifest['batches']:
        chunk_path = os.path.join(CACHE_DIR, f'chunk_{batch_entry["chunk"]:02d}_results.csv')
        if os.path.exists(chunk_path):
            print(f"\n  Chunk {batch_entry['chunk']}: Already downloaded")
            continue

        batch = client.messages.batches.retrieve(batch_entry['batch_id'])
        batch_entry['status'] = batch.processing_status

        if batch.processing_status == 'ended':
            download_chunk(batch_entry, api_key, client)
            batch_entry['status'] = 'downloaded'
        else:
            print(f"\n  Chunk {batch_entry['chunk']} ({batch_entry['batch_id']}): "
                  f"{batch.processing_status} "
                  f"({batch.request_counts.succeeded:,} succeeded, "
                  f"{batch.request_counts.processing:,} processing)")

    save_manifest(manifest)

    # Step 1b: Patch any chunks with retry results
    retry_files = [f for f in os.listdir(CACHE_DIR) if f.startswith('retry_chunk_') and f.endswith('.json')]
    if retry_files:
        print("\n" + "=" * 60)
        print("STEP 1b: Patch chunks with retry results")
        print("=" * 60)
        for rf in sorted(retry_files):
            chunk_num = int(rf.replace('retry_chunk_', '').replace('.json', ''))
            patch_chunk_with_retry(client, CACHE_DIR, chunk_num)

    # Step 2: Submit pending chunks
    pending = manifest.get('pending_chunks', [])
    if pending:
        print("\n" + "=" * 60)
        print(f"STEP 2: Submit {len(pending)} pending chunks")
        print("=" * 60)

        # Load merged dataset for building requests
        print(f"\n  Loading merged dataset...")
        merged_df = pd.read_csv(MERGED_PATH)

        submitted_this_run = 0
        still_pending = []

        for chunk_entry in pending:
            try:
                new_batch = submit_chunk(chunk_entry, api_key, client, merged_df)
                manifest['batches'].append(new_batch)
                submitted_this_run += 1
            except Exception as e:
                error_msg = str(e)
                if '429' in error_msg or 'rate' in error_msg.lower():
                    print(f"  Rate limited. Stopping submissions. {len(pending) - submitted_this_run - len(still_pending)} chunks deferred.")
                    still_pending.append(chunk_entry)
                    still_pending.extend(pending[pending.index(chunk_entry) + 1:])
                    break
                else:
                    print(f"  Error submitting chunk {chunk_entry['chunk']}: {e}")
                    still_pending.append(chunk_entry)

        manifest['pending_chunks'] = still_pending
        save_manifest(manifest)
        print(f"\n  Submitted {submitted_this_run} chunks this run")
        if still_pending:
            print(f"  {len(still_pending)} chunks still pending")
    else:
        print("\n  No pending chunks to submit.")

    # Step 3: Check if all chunks are downloaded
    print("\n" + "=" * 60)
    print("STEP 3: Check completion status")
    print("=" * 60)

    total_expected_chunks = (manifest['total_records'] + manifest['chunk_size'] - 1) // manifest['chunk_size']
    downloaded = sum(1 for i in range(1, total_expected_chunks + 1)
                     if os.path.exists(os.path.join(CACHE_DIR, f'chunk_{i:02d}_results.csv')))

    print(f"\n  Chunks downloaded: {downloaded}/{total_expected_chunks}")

    if downloaded == total_expected_chunks:
        print("\n  All chunks complete! Merging...")
        merge_all_chunks(manifest)
    else:
        pending_count = len(manifest.get('pending_chunks', []))
        in_progress = total_expected_chunks - downloaded - pending_count
        print(f"  In progress: {in_progress}")
        print(f"  Pending submission: {pending_count}")
        print(f"\n  Run this script again later to check on progress and submit remaining chunks.")


if __name__ == '__main__':
    main()
