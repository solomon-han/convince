# convince/allennlp/allennlp/data/dataset_readers/reading_comprehension/ __init__.py
"""
Reading comprehension is loosely defined as follows: given a question and a passage of text that
contains the answer, answer the question.

These submodules contain readers for things that are predominantly reading comprehension datasets.
"""

from allennlp.data.dataset_readers.reading_comprehension.babi import BabiSingleReader
from allennlp.data.dataset_readers.reading_comprehension.dream_mc import DreamMCReader
from allennlp.data.dataset_readers.reading_comprehension.squad import SquadReader
from allennlp.data.dataset_readers.reading_comprehension.quac import QuACReader
from allennlp.data.dataset_readers.reading_comprehension.race_mc import RaceMCReader
from allennlp.data.dataset_readers.reading_comprehension.triviaqa import TriviaQaReader
from allennlp.data.dataset_readers.reading_comprehension.qangaroo import QangarooReader
from allennlp.data.dataset_readers.reading_comprehension.hotpot_rc_reader import HotpotRCReader
