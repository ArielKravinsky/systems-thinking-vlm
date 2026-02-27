import random
from pathlib import Path
from typing import List, Dict, Optional, Iterator, Tuple
from PIL import Image


class SystemsThinkingDataset:
    """DataLoader for systems thinking dataset with Hebrew questions/answers.
    
    Structure:
    - questions/<question_num>.txt
    - images/<question_num>_<answer_num>.<ext>
    - answers/<question_num>_<answer_num>_<participant_num>.txt
    """
    
    def __init__(self, root: Path):
        self.root = Path(root)
        self.images_dir = self.root / "images"
        self.answers_dir = self.root / "answers"
        self.questions_dir = self.root / "questions"
        
        # Build dataset index
        self.samples = []
        self._index_dataset()
    
    def _index_dataset(self):
        """Index all valid image/answer pairs with their questions."""
        skipped = []

        for ans_path in sorted(self.answers_dir.glob("*.txt")):
            stem = ans_path.stem  # expected: question_answer_participant
            parts = stem.split('_')
            if len(parts) != 3:
                skipped.append((stem, "invalid answer filename format"))
                continue

            question_num, answer_num, participant_num = parts
            q_path = self.questions_dir / f"{question_num}.txt"

            image_path = None
            for ext in ['.png', '.jpg', '.jpeg', '.bmp', '.gif', '.webp']:
                candidate = self.images_dir / f"{question_num}_{answer_num}{ext}"
                if candidate.exists():
                    image_path = candidate
                    break

            if not q_path.exists():
                skipped.append((stem, f"missing question: {q_path.name}"))
                continue
            if image_path is None:
                skipped.append((stem, f"missing image: {question_num}_{answer_num}.*"))
                continue

            question_text = q_path.read_text(encoding='utf-8').strip()
            answer_text = ans_path.read_text(encoding='utf-8').strip()
            if not question_text:
                skipped.append((stem, f"empty question: {q_path.name}"))
                continue
            if not answer_text:
                skipped.append((stem, f"empty answer: {ans_path.name}"))
                continue

            self.samples.append({
                'question_num': question_num,
                'answer_num': answer_num,
                'participant_num': participant_num,
                'stem': stem,
                'image_path': image_path,
                'answer_path': ans_path,
                'question_path': q_path,
            })
        
        # Log skipped samples
        if skipped:
            print(f"[WARN] Skipped {len(skipped)} incomplete samples:")
            for stem, reason in skipped:
                print(f"  - {stem}: {reason}")
    
    def __len__(self) -> int:
        return len(self.samples)
    
    def __getitem__(self, idx: int) -> Dict:
        """Get a single sample by index."""
        sample = self.samples[idx]
        return {
            'question_num': sample['question_num'],
            'answer_num': sample['answer_num'],
            'participant_num': sample['participant_num'],
            'stem': sample['stem'],
            'image': Image.open(sample['image_path']).convert('RGB'),
            'image_path': sample['image_path'],
            'question': sample['question_path'].read_text(encoding='utf-8').strip(),
            'answer': sample['answer_path'].read_text(encoding='utf-8').strip(),
        }
    
    def sample(self, n: int = 1) -> List[Dict]:
        """Randomly sample n items from the dataset."""
        indices = random.sample(range(len(self)), min(n, len(self)))
        return [self[i] for i in indices]
    
    def __iter__(self) -> Iterator[Dict]:
        """Iterate over all samples."""
        for i in range(len(self)):
            yield self[i]
    
    def get_batch(self, batch_size: int, start_idx: int = 0) -> List[Dict]:
        """Get a batch of samples starting from start_idx."""
        end_idx = min(start_idx + batch_size, len(self))
        return [self[i] for i in range(start_idx, end_idx)]
    
    def batches(self, batch_size: int) -> Iterator[List[Dict]]:
        """Iterate in batches."""
        for i in range(0, len(self), batch_size):
            yield self.get_batch(batch_size, i)
