import fire

from constants import DATA_DIR, DUMP_DIR
from knowledge_embed import KnowledgeData
from raw_data_fix import RawData
from utils import check_input_dir, check_output_dir


class CommandLineInterface:
    def preprocess(self):
        """Preprocess the test_data in the DATA_DIR and dump the results into the DUMP_DIR."""

        # Check if the required test_data files exist.
        check_input_dir(DATA_DIR,
                        ['dialogs_now/train/', 'dialogs_now/valid/', 'dialogs_now/test/', 'knowledge/products/'],
                        ['url2img.txt', 'knowledge/styletip/styletips_synset.txt',
                         'knowledge/celebrity/celebrity_distribution.json'])
        # Check if the dump directory exists.
        check_output_dir(DUMP_DIR)

        raw_data = RawData()  # Get raw test_data from the dialogs.
        KnowledgeData(raw_data)  # Get knowledge test_data (style tips, celebrity popularity, product attributes).

    def train_text(self):
        from trainer_text import TrainerText

        check_input_dir(DATA_DIR, ['images/', 'knowledge/products/'], [])

        trainer = TrainerText()
        trainer.train()

    def train_dgl(self):
        from trainer_dgl import TrainerDGL

        trainer = TrainerDGL()
        trainer.train()

    def eval_graph(self):
        from evaluator_graph import Evaluator

        evaluator = Evaluator()
        evaluator.eval()

    def eval_text(self):
        from evaluator_text import Evaluator

        evaluator = Evaluator()
        evaluator.eval()

    def eval_case(self):
        from evaluator_graph_case import Evaluator

        evaluator = Evaluator()
        evaluator.eval()

    def eval_text_case(self):
        from evaluator_text_case import Evaluator

        evaluator = Evaluator()
        evaluator.eval()


if __name__ == '__main__':
    fire.Fire(CommandLineInterface)
