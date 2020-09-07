{
  "dataset_reader": {
    "type": "hotpot-rc",
    "token_indexers": {
      "tokens": {
          "type": "bert-pretrained",
          "pretrained_model": "datasets/bert/uncased_L-12_H-768_A-12/vocab.txt",
          "do_lowercase": true,
          "use_starting_offsets": true
      }
    }
  },
  "train_data_path": "/content/drive/My Drive/HotpotQA/training.json",
  "validation_data_path": "/content/drive/My Drive/HotpotQA/dev_distractor.json",
  "model": {
    "type": "bert-mc-gpt",
    "text_field_embedder": {
      "allow_unmatched_keys": true,
      "embedder_to_indexer_map": {
        "tokens": ["tokens", "tokens-offsets", "token-type-ids"]
      },
      "token_embedders": {
        "tokens": {
          "type": "bert-pretrained",
          "pretrained_model": "bert-base-uncased",
          "requires_grad": true,
          "top_layer_only": true
        }
      }
    }
  },
  "iterator": {
    "type": "bucket",
    "sorting_keys": [["passage", "num_tokens"], ["question", "num_tokens"]],
    "batch_size": 1
  },

  "trainer": {
    "num_epochs": 20,
    "patience": 3,
    "validation_metric": "+start_acc",
    "cuda_device": 0,
    "learning_rate_scheduler": {
      "type": "reduce_on_plateau",
      "factor": 0.67,
      "mode": "max",
      "patience": 1
    },
    "optimizer": {
      "lr": 0.00001,
      "type": "bert_adam"
    }
  }
}
