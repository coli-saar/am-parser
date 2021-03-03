# config file for toy data set (while implementing/debugging)
local epochs = 10;
local cudadevice = 0;  # -1

local pos_dim = 32;
# local pos_dim = 0;
# local src_type_dim = 0;
local src_type_dim = 64;
local token_dim = 50;
local encoder_input_dim = token_dim + src_type_dim + pos_dim;

# if dimension is 0, defaults to null --> no embedding used for this part
local pos_emb = if pos_dim > 0 then {embedding_dim: pos_dim, vocab_namespace: "src_pos"};
local src_type_emb = if src_type_dim > 0 then {embedding_dim: src_type_dim, vocab_namespace: "src_types"};

# in train/valid path and dataset_reader init parameter:
# insertion point for source_target_foldername_pair
local splitmarker = '@@SPLITMARKER@@';

{
  dataset_reader: {
    type: 'typeamconllreader', # see typeamconllreader.py
    lazy: false,
    source_token_indexers: {
      xtokens: {  # debug: called it xtokens to see where it is referenced
        type: 'single_id',
        namespace: 'src_words'
      }
    },
    source_target_foldername_pair: ['PAS', 'DM'], # normal direction
    # source_target_foldername_pair: ['DM', 'PAS'], # reverse direction
    splitmarker: splitmarker,
  },
  train_data_path: 'typetagging/toydata/' + splitmarker + '/train/train.amconll',
  validation_data_path: 'typetagging/toydata/' + splitmarker + '/gold-dev/gold-dev.amconll',
  model: {
    type: 'typetaggingmodel', # see corresponding py class and file
    text_field_embedder: {  # param of TypeTaggingModel.__init__
      token_embedders: {
        xtokens: {
        type: 'embedding',  # <<----todo--->> embedding here
          vocab_namespace: "src_words",
          # pretrained_file: "(http://nlp.stanford.edu/data/glove.6B.zip)#glove.6B.50d.txt",
          embedding_dim: token_dim,
          # trainable: false
        }
      }
    },
    pos_tag_embedding: pos_emb,
    src_type_embedding: src_type_emb,
    encoder: { # param of TypeTaggingModel.__init__
      type: 'lstm',
      input_size: encoder_input_dim,
      hidden_size: 50,
      num_layers: 1,
      bidirectional: true
    }
  },
  data_loader: {
    batch_sampler: {
      type: 'bucket',  # try to put instances with similar length in same batch
      batch_size: 10
      # "padding_noise": 0.0 ??
    }
  },
  trainer: {
    num_epochs: epochs,
    patience: 3,  # used for early stopping
    validation_metric: '-loss',  # used for early stopping
    # or "+sequence_accuracy" ?
    cuda_device: cudadevice,
    grad_clipping: 5.0,  # todo: do we need gradient clipping?
    optimizer: {  # just using some default here
      type: 'adam',
      lr: 0.01
    }
  }
}