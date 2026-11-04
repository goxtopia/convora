"""Source-only checks: tokenizer contract, FIFO/cancel behavior and model identity.

Requires tokenizers. No weight files, sockets, ADB or training.
"""
import json,sys,unittest
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'runtime/pico-chat'))
from server import PromptBuilder,Scheduler,CONFIG

class SourceChecks(unittest.TestCase):
    def test_template_golden(self):
        golden=json.loads((ROOT/'runtime/pico-chat/template-golden.json').read_text(encoding='utf-8'))
        ids,meta=PromptBuilder().build(golden['messages'])
        self.assertEqual(ids,golden['prefix_ids']);self.assertEqual(meta['retained_messages'],3)

    def test_fifo_capacity_and_cancel(self):
        queue=Scheduler(2);first=queue.enqueue();cancelled=queue.enqueue()
        self.assertIsNone(queue.enqueue());self.assertFalse(queue.start(cancelled));self.assertTrue(queue.start(first))
        queue.release(cancelled);next_ticket=queue.enqueue();self.assertEqual(queue.position(next_ticket),1)
        queue.release(first);self.assertTrue(queue.start(next_ticket));queue.release(next_ticket)
        self.assertEqual(queue.status(),{'busy':False,'waiting':0,'capacity':2})

    def test_formal_model_identity(self):
        model=json.loads((ROOT/'models/model.json').read_text(encoding='utf-8'))
        self.assertEqual(CONFIG,model)
        artifacts=json.loads((ROOT/'ARTIFACTS.json').read_text(encoding='utf-8'))['files']
        self.assertEqual(model['rknn_sha256'],artifacts['models/rknn/model.rknn']['sha256'])
        self.assertEqual(model['context_tokens'],512)

if __name__=='__main__':unittest.main()
