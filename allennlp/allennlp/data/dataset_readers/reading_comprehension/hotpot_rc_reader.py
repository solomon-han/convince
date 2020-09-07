#Hotpot Reader ver1
#convince/allennlp/allennlp/data/dataset_readers/reading_comprehension/hotpot_rc_reader.py
import json
import logging
import os
from typing import Dict, List

from overrides import overrides

from allennlp.common.file_utils import cached_path
from allennlp.data.fields import Field
from allennlp.data.dataset_readers.dataset_reader import DatasetReader
from allennlp.data.fields import TextField, IndexField, MetadataField, ListField
from allennlp.data.instance import Instance
from allennlp.data.token_indexers import SingleIdTokenIndexer, TokenIndexer
from allennlp.data.tokenizers import Token, Tokenizer, WordTokenizer

logger = logging.getLogger(__name__)  # pylint: disable=invalid-name


@DatasetReader.register("hotpot-rc")
class HotpotRCReader(DatasetReader):
    """
    Reads a JSON-formatted Race file and returns a ``Dataset`` where the ``Instances`` have four
    fields: ``question``, a ``TextField``, ``passage``, another ``TextField``, and ``span_start``
    and ``span_end``, both ``IndexFields`` into the ``passage`` ``TextField``.  We also add a
    ``MetadataField`` that stores the instance's ID, the original passage text, gold answer strings,
    and token offsets into the original passage, accessible as ``metadata['id']``,
    ``metadata['original_passage']``, ``metadata['answer_texts']`` and
    ``metadata['token_offsets']``.  This is so that we can more easily use the official SQuAD
    evaluation script to get metrics.

    Parameters
    ----------
    tokenizer : ``Tokenizer``, optional (default=``WordTokenizer()``)
        We use this ``Tokenizer`` for both the question and the passage.  See :class:`Tokenizer`.
        Default is ```WordTokenizer()``.
    token_indexers : ``Dict[str, TokenIndexer]``, optional
        We similarly use this for both the question and the passage.  See :class:`TokenIndexer`.
        Default is ``{"tokens": SingleIdTokenIndexer()}``.
    """
    def __init__(self,
                 tokenizer: Tokenizer = None,
                 token_indexers: Dict[str, TokenIndexer] = None,
                 lazy: bool = True,
                 debate_mode: List[str] = None) -> None:
        super().__init__(lazy)
        self._tokenizer = tokenizer or WordTokenizer()
        self._token_indexers = token_indexers or {'tokens': SingleIdTokenIndexer()}
        self._letter_to_answer_idx = {'A': 0, 'B': 1, 'C': 2, 'D': 3}
        self._debate_modes = {
            ('e',): [['ⅰ'], ['ⅱ'], ['ⅲ'], ['ⅳ']],
            ('E',): [['Ⅰ'], ['Ⅱ'], ['Ⅲ'], ['Ⅳ']],
        }.get(tuple(debate_mode), [None])

    @staticmethod
    def _filepath_to_id(filepath: str, q_no: int) -> str:
        file_parts = os.path.join(filepath, str(q_no)).split('/')[2:]  # NB: [2:] -> [-4:] (to make more general)
        for split in ['train', 'dev', 'test']:
            if split in file_parts[0]:
                file_parts[0] = split
            elif file_parts[0] in {'A', 'B', 'C', 'D', 'E'}:
                file_parts[0] = 'test'  # Question-type datasets come from test
        return '/'.join(file_parts)

    @overrides
    def _read(self, file_path: str):
        # if `file_path` is a URL, redirect to the cache
        file_path = cached_path(file_path)

        logger.info("Reading files at %s", file_path)

        # Load json file, Get all articles
        with open(file_path, 'r') as f:
                qa_data_list = json.load(f)

        for qa_data in qa_data_list:

            # Article-level info (concatenate all passages)
            title = qa_data["_id"]
            if len(qa_data["context"]) < 1: # Exception Handling
              continue
            passage_list = []
            for title_sentlist_pair in qa_data["context"]:
              if len(title_sentlist_pair)!=2: # Exception Handling
                continue
              _, sentlist = title_sentlist_pair
              passage = " ".join(sentlist)
              passage_list.append(passage)

            passage_text = " ".join(passage_list)
            tokenized_passage = self._tokenizer.tokenize(passage_text)

            # inline utility function
            def base_idx_dict(context):
              idx = 0
              output = {}
              for title, passage in context:
                output[title] = idx
                idx += len(passage)
              return output

            def to_absolute_idx(base_dict, title, idx):
              if title in base_dict.keys():
                base = base_dict[title]
              else:
                # Report Exception
                print('KEY EXCEPTION: ',title,'/',idx,' IN')
                print(base_dict.keys())
                return -1 # Exception
              return base + idx

            # Process evidence sentence labels
            base_dict = base_idx_dict(qa_data["context"])
            ev_idx_label_list = []
            for ev in qa_data["supporting_facts"]:
              absolute_idx = to_absolute_idx(base_dict, ev[0], ev[1])
              if absolute_idx == -1:
                continue # Exception
              ev_idx_label_list.append(absolute_idx)

            # Iterate through questions (one option = right answer)
            question_text = qa_data["question"].strip().replace("\n", "")
            options_text = [qa_data["answer"], "dummy"]
            answer_index = 0
            qid = title
            for debate_mode in self._debate_modes:
                yield self.text_to_instance(question_text, passage_text, options_text, qid, tokenized_passage,
                                                    answer_index, ev_idx_label_list, debate_mode)

    @overrides
    def text_to_instance(self,  # type: ignore
                         question_text: str,
                         passage_text: str,
                         options_text: List[str],
                         qa_id: str,
                         passage_tokens: List[Token] = None,
                         answer_index: int = None,
                         ev_idx_label_list:  List[int] = None,
                         debate_mode: List[str] = None) -> Instance:
        # pylint: disable=arguments-differ
        additional_metadata = {'id': qa_id, 'debate_mode': debate_mode}
        fields: Dict[str, Field] = {}

        # Tokenize and calculate token offsets (i.e., for wordpiece)
        question_tokens = self._tokenizer.tokenize(question_text)
        if passage_tokens is None:
            passage_tokens = self._tokenizer.tokenize(passage_text)
        options_tokens = self._tokenizer.batch_tokenize(options_text)
        passage_offsets = [(token.idx, token.idx + len(token.text)) for token in passage_tokens]

        # This is separate so we can reference it later with a known type.
        options_field = ListField([TextField(option_tokens, self._token_indexers) for option_tokens in options_tokens])
        fields['passage'] = TextField(passage_tokens, self._token_indexers)
        fields['question'] = TextField(question_tokens, self._token_indexers)
        fields['options'] = options_field
        metadata = {'original_passage': passage_text, 'token_offsets': passage_offsets,
                    'question_tokens': [token.text for token in question_tokens],
                    'passage_tokens': [token.text for token in passage_tokens],
                    'options_tokens': [[token.text for token in option_tokens] for option_tokens in options_tokens],
                    'ev_idx_label_list':ev_idx_label_list}
        if answer_index is not None:
            metadata['answer_texts'] = options_text[answer_index]

        fields['answer_index'] = IndexField(answer_index, options_field)

        metadata.update(additional_metadata)
        fields['metadata'] = MetadataField(metadata)
        return Instance(fields)
