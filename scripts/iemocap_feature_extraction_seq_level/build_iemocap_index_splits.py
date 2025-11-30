# iemocap_data_processor.py

import os
import re
import pandas as pd
from pathlib import Path
from glob import glob
from tqdm import tqdm
import sys

# ============================ Configuration & Paths ============================

# Assumed base path is consistent with the notebook
BASE = '../data/IEMOCAP_full_release'
SESSIONS = [s for s in os.listdir(BASE) if s.lower().startswith('session')]
DATA_OUTPUT_DIR = Path('../data') # CSV export path

# Regex for parsing the EmoEvaluation file headers
HEAD_RE = re.compile(
    r'^\[(\d+\.\d+)\s*-\s*(\d+\.\d+)\]\s+(\S+)\s+([A-Za-z]{3})\s*\[\s*([\d\.\s,]+)\s*\]\s*$'
)

# 6 Core category selection and mapping, consistent with the notebook
KEEP_6 = {'ang','hap','sad','neu','fru','exc'}
NAME_MAP_6 = {'ang':'angry', 'hap':'happy', 'sad':'sad', 'neu':'neutral', 'fru':'frustration', 'exc':'excited'}

# ============================ Utility Functions ============================

def read_text_robust(p: Path) -> str:
    """Tries several encodings; falls back to ignoring errors."""
    for enc in ('utf-8', 'utf-8-sig', 'cp1252', 'latin-1'):
        try:
            with open(p, 'r', encoding=enc) as f:
                return f.read()
        except UnicodeDecodeError:
            continue
    # last resort
    with open(p, 'r', encoding='utf-8', errors='ignore') as f:
        return f.read()

def uid_from_stem(stem: str):
    """Extracts Utterance ID from filename stem (e.g., "Ses01F_impro01_F000")"""
    m = re.search(r'(Ses\d{2}[MF]_[A-Za-z]+\d+_[MF]\d{3,4})', stem)
    return m.group(1) if m else None

def dialog_id_from_uid(uid: str):
    """Extracts Dialog ID from Utterance ID (e.g., "Ses01F_impro01_F000" -> "Ses01F_impro01")"""
    m = re.match(r'^(Ses\d{2}[MF]_[A-Za-z]+\d+)_', uid)
    return m.group(1) if m else None

def clean_transcript_line(line: str) -> str:
    """Cleans transcript lines of IDs, timestamps, noise, and special tags."""
    # Remove ID and timestamps
    line = re.sub(r'^Ses\d{2}[MF]_[A-Za-z]+\d+_[MF]\d{3,4}\s*\[.*?\]\s*:\s*', '', line)
    # Clean noise and special tags
    line = re.sub(r'</?s>', ' ', line, flags=re.I)
    line = re.sub(r'<\s*(sil|sp|noise|laughter)\s*>', ' ', line, flags=re.I)
    line = re.sub(r'\(\d+\)', ' ', line)
    line = re.sub(r'<[^>]+>', ' ', line)
    # Normalize whitespace
    return re.sub(r'\s+', ' ', line).strip()

def text_from_transcript_file(path: str, utter_id: str) -> str | None:
    """Extracts matching text from the transcript file."""
    pat = re.compile(rf'^{re.escape(utter_id)}\b')
    try:
        with open(path, 'r', encoding='utf-8', errors='ignore') as f:
            for raw in f:
                if pat.search(raw):
                    return clean_transcript_line(raw)
    except FileNotFoundError:
        return None
    return None

def wdseg_to_text(path: str) -> str:
    """Converts forced-alignment file (.wdseg) to plain text (used as backup)."""
    sent = []
    try:
        with open(path, 'r', encoding='utf-8', errors='ignore') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 3:
                    w = parts[-1].lower()
                    if w in {'sil','sp','garbage','<sil>','</s>','<s>'}:
                        continue
                    w = re.sub(r'[{}<>]', '', w)
                    if w:
                        sent.append(w)
    except FileNotFoundError:
        return ""
    return ' '.join(sent)

# ============================ Main Processing Functions ============================

def parse_emotion_labels() -> pd.DataFrame:
    """Parses all EmoEvaluation files to extract labels."""
    print("--- 1. Parsing Emotion Label Files ---")
    label_rows = []
    
    # Sort sessions naturally
    sorted_sessions = sorted([s for s in SESSIONS if re.search(r'\d+', s)], 
                             key=lambda x: int(re.search(r'\d+', x).group()))

    for S in tqdm(sorted_sessions, desc="Parsing Labels"):
        emo_dir = Path(BASE, S, 'dialog', 'EmoEvaluation')
        if not emo_dir.is_dir():
            continue
        for emo_file in emo_dir.glob('*.txt'):
            content = read_text_robust(emo_file)
            for line in content.splitlines():
                m = HEAD_RE.match(line.strip())
                if m:
                    start, end, utt, lab, vad = m.groups()
                    label_rows.append({
                        'session': S,
                        'utter_id': utt,
                        'label_raw': lab.lower(),
                        't_start': float(start),
                        't_end': float(end),
                        'vad': vad,
                        'emo_file': str(emo_file)
                    })
    
    labels_df = pd.DataFrame(label_rows)
    print(f"Original labels parsed: {len(labels_df)} records.")
    
    # Filter and map to the 6 core categories
    labels6 = labels_df[labels_df['label_raw'].isin(KEEP_6)].copy()
    labels6['label'] = labels6['label_raw'].map(NAME_MAP_6)
    
    print(f"Filtered {len(labels6)} core {len(KEEP_6)} category records.")
    return labels6[['session', 'utter_id', 'label']]

def build_global_index(labels_df: pd.DataFrame) -> pd.DataFrame:
    """Builds the global index, aligning audio, text, and labels."""
    print("\n--- 2. Building Global Index and Aligning Text ---")
    rows = []
    
    # Sort sessions naturally
    sorted_sessions = sorted([s for s in SESSIONS if re.search(r'\d+', s)], 
                             key=lambda x: int(re.search(r'\d+', x).group()))

    for S in tqdm(sorted_sessions, desc="Aligning Paths"):
        Sdir = Path(BASE, S)

        # 1. Path Mapping
        wavs = [Path(p) for p in glob(str(Sdir/'sentences'/'**'/'*.wav'), recursive=True)]
        wav_map = {uid_from_stem(p.stem): str(p) for p in wavs if uid_from_stem(p.stem)}

        wdsegs = [Path(p) for p in glob(str(Sdir/'sentences'/'ForcedAlignment'/'**'/'*.wdseg'), recursive=True)]
        wdseg_map = {uid_from_stem(p.stem): str(p) for p in wdsegs if uid_from_stem(p.stem)}

        dial_txts = [Path(p) for p in glob(str(Sdir/'dialog'/'transcriptions'/'**'/'*.txt'), recursive=True)]
        txt_map = {p.stem: str(p) for p in dial_txts}

        # 2. Combine all path info
        for uid in sorted(set(wav_map) | set(wdseg_map)):
            did = dialog_id_from_uid(uid)
            rows.append({
                'session': S,
                'utter_id': uid,
                'audio_path': wav_map.get(uid),
                'wdseg_path': wdseg_map.get(uid),
                'transcript_path': txt_map.get(did) if did else None,
            })
    
    index_df = pd.DataFrame(rows)

    # 3. Label Merge
    merged = index_df.merge(labels_df, on=['session','utter_id'], how='inner')
    print(f"Paths and labels merged: {len(merged)} records.")

    # 4. Text Extraction
    def pick_text(row):
        # Prefer transcript file, fall back to wdseg
        if pd.notna(row['transcript_path']):
            txt = text_from_transcript_file(row['transcript_path'], row['utter_id'])
            if txt:
                return txt
        if pd.notna(row['wdseg_path']):
            return wdseg_to_text(row['wdseg_path'])
        return None

    merged['text'] = merged.apply(pick_text, axis=1)

    # 5. Final Filtering and Cleanup
    final_df = merged[(merged['audio_path'].notna()) & (merged['text'].notna())].copy()
    final_df = final_df.sort_values(['session','utter_id']).reset_index(drop=True)

    # Convert paths to absolute paths (Recommended)
    for col in ['audio_path', 'wdseg_path', 'transcript_path']:
        final_df[col] = final_df[col].apply(lambda p: str(Path(p).resolve()) if pd.notna(p) else p)

    print(f"Total Final Aligned Samples (Text + Audio + Label): {len(final_df)}")
    return final_df

def split_and_export(final_df: pd.DataFrame, output_dir: Path):
    """Splits the dataset by standard Session rules and exports CSVs."""
    print("\n--- 3. Splitting Dataset and Exporting ---")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 1. Save preliminary index
    index_cols = ['utter_id','session','audio_path','text','label']
    index_path = output_dir / 'iemocap_index.csv'
    
    final_df[index_cols].to_csv(index_path, index=False)
    print(f"1/2 Preliminary index saved to: {index_path}")

    # 2. Split dataset (Session-based split to prevent speaker overlap)
    # Session 5 -> test; Session 4 -> val; Others -> train
    split_map = {'Session5': 'test', 'Session4': 'val'}
    final_df['split'] = final_df['session'].map(split_map).fillna('train')

    # 3. Export the complete index with split information
    split_index_path = output_dir / 'iemocap_index_splits.csv'
    final_df.to_csv(split_index_path, index=False)
    print(f"2/2 Complete index (with splits) saved to: {split_index_path}")

    # 4. Display final statistics
    print("\n" + "="*40)
    print("Dataset Split Statistics (IEMOCAP Standard Split)")
    print("="*40)
    
    print("\nSplit Distribution:")
    print(final_df['split'].value_counts())

    print("\nLabel Counts per Split:")
    label_counts = final_df.groupby(['split', 'label']).size().reset_index(name='Count')
    print(label_counts.to_string(index=False))
    print("="*40)


# ============================ Execution Block ============================

if __name__ == "__main__":
    # Ensure BASE path is correct before starting
    if not Path(BASE).is_dir():
        print(f"FATAL ERROR: IEMOCAP raw data directory not found at: {BASE}")
        sys.exit(1)

    # 1. Parse Labels
    labels_df_6 = parse_emotion_labels()
    
    # 2. Build Index and Extract Text
    final_df = build_global_index(labels_df_6)
    
    # 3. Split and Export
    split_and_export(final_df, DATA_OUTPUT_DIR)
    
    print("\nAll steps completed. CSV index files generated.")