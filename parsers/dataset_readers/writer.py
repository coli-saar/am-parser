from typing import TextIO, Iterable, Dict, Any

from allennlp.data import Vocabulary
from allenpipeline import DatasetWriter

from topdown_parser.dataset_readers.amconll_tools import AMSentence


@DatasetWriter.register("amconll_writer")
class AMConllWriter(DatasetWriter):

    def write_to_file(self, vocab : Vocabulary, instances: Iterable[Dict[str, Any]], file : TextIO) -> None:
        """
        Write instances coming either from model.decode() or from your own BatchDecoder to a file.
        DON'T close the file in this method.
        :param vocab:
        :param instances:
        :param file:
        :return:
        """
        for inst in instances:
            am_pred : AMSentence = inst["predictions"]
            file.write(str(am_pred))
            file.write("\n\n")