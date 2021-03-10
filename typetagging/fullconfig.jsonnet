# config file for full laptop-based data set (while implementing/debugging)
# allennlp train -f --include-package typetagging -s ./typetagging/tmp/full ./typetagging/fullconfig.jsonnet
# tensorboard --logdir ./typetagging/tmp/full
# allennlp evaluate --include-package typetagging ./typetagging/tmp/full/model.tar.gz ~/HiwiAK/data/sempardata/ACL2019/SemEval/2015/@@SPLITMARKER@@/gold-dev/gold-dev.amconll
#
# todo: change from allennlp >1 to 0.8 mentioned in readme?
# todo: include pretrained contextualized embeddings (bert? elmo?)
# todo: add more dropout maybe? (input, encoder, classifier ...)

local epochs = 15;
local patience = 2;
local cudadevice = 0;  # -1

local include_pos = true;
local include_srctypes = true;

local pos_dim = if include_pos then 32 else 0;
local src_type_dim = if include_srctypes then 32 else 0;

# glove
local glove_dim = 200;
local glove_file = "/home/wurzel/HiwiAK/data/pretrained-embeddings/glove/"+"glove.6B.200d.txt";
# local glove_file = "(http://nlp.stanford.edu/data/glove.6B.zip)#glove.6B.200d.txt"
# bert # todo untested
# local bert_model = "bert-base-uncased";
# local bert_dim = 768;  # bert-base-uncased: 768

# local token_dim = 64;
# local encoder_input_dim = bert_dim + src_type_dim + pos_dim;  # ignores token emb!
local encoder_input_dim = glove_dim + src_type_dim + pos_dim;  # todo ignores token emb!
# local encoder_input_dim = glove_dim + token_dim + src_type_dim + pos_dim;
# local encoder_input_dim = token_dim + src_type_dim + pos_dim;

# if dimension is 0, defaults to null --> no embedding used for this part
local pos_emb = if include_pos then {embedding_dim: pos_dim, vocab_namespace: "src_pos"};
local src_type_emb = if include_srctypes then {embedding_dim: src_type_dim, vocab_namespace: "src_types"};

local encoder_hidden_size = 128;  # 64

# in train/valid path and dataset_reader init parameter:
# insertion point for source_target_foldername_pair
local splitmarker = '@@SPLITMARKER@@';

{
  dataset_reader: {
    type: 'typeamconllreader', # see typeamconllreader.py
    lazy: false,
    source_token_indexers: {
      # tokens: { type: 'single_id', namespace: 'src_words'},
      # glove start
      "glove": {
         type: "single_id",
         lowercase_tokens: true,
         namespace: 'src_words'  # todo: needed?
      },
      # glove end
      # bert start  # todo: untested bert
      # "bert": {
      #  "type": "pretrained_transformer_mismatched",  # mismatched=didn't use bert tokenizer
      #  "model_name": bert_model,
      #  "namespace": 'src_words'  # needed?
      #  #"do_lowercase": true
      # }
      # bert end
    },
    # source_target_foldername_pair: ['PAS', 'PSD'],  # x
    source_target_foldername_pair: ['PAS', 'DM'],  # normal
    # source_target_foldername_pair: ['DM', 'PAS'],  # reversed
    splitmarker: splitmarker,
  },
  train_data_path: '/home/wurzel/HiwiAK/data/sempardata/ACL2019/SemEval/2015/' + splitmarker + '/train/train.amconll',
  validation_data_path: '/home/wurzel/HiwiAK/data/sempardata/ACL2019/SemEval/2015/' + splitmarker + '/gold-dev/gold-dev.amconll',
  model: {
    type: 'typetaggingmodel', # see corresponding py class and file
    # below follow parameters of TypeTaggingModel.__init__
    text_field_embedder: {
      token_embedders: {
        # tokens: {
        #   type: 'embedding',
        #   vocab_namespace: "src_words",
        #   embedding_dim: token_dim,
        #   # trainable: false
        # },
        # glove start
        "glove": {
          type: "embedding",
          vocab_namespace: "src_words",  # todo: needed?
          embedding_dim: glove_dim,
          pretrained_file: glove_file,
          trainable: true  # todo trainable ?
        },
        # glove end
        # bert start # todo bert untested
        # "bert": {
        #     "type": "pretrained_transformer_mismatched", # mismatched: not tokenized using bert's tokenizer
        #     "model_name": bert_model,
        #     #"vocab_namespace": "src_words",  # needed?
        #     "train_parameters": false
        # },
        # bert end
      },
    },
    pos_tag_embedding: pos_emb,
    src_type_embedding: src_type_emb,
    encoder: {
      # type: 'stacked_bidirectional_lstm',  # -> StackedBidirectionalLstm
      type: 'lstm',  # -> LstmSeq2SeqEncoder, actually a bilstm, see below
      # todo: influence of type on whether need *2 for bidirectionality elsewhere?
      num_layers: 2,
      input_size: encoder_input_dim,
      hidden_size: encoder_hidden_size,
      # --would require: lstm   as type:
      # parameters from allennlp.modules.LstmSeq2SeqEncoder (pytorch wrapper)
      bidirectional: true,
      dropout: 0.3, # default: 0
      # -- would require: stacked_bidirectional_lstm  as type:
      # parameters from allennlp.modules.StackedBiDirectionalLSTM: (slower)
      # recurrent_dropout_probability: 0.0, # float, optional (default = 0.0)
      # layer_dropout_probability: 0.0, # float, optional (default = 0.0)
      # use_highway: bool, optional (default = True)
    },
    classifier: {
      # parameter names from allennlp.modules.FeedForward:
      input_dim: encoder_hidden_size * 2, # *2 for bidirectionality of lstm
      num_layers: 1,
      hidden_dims: [128],
      activations: ['relu'],
      # dropout: Union[float, List[float]] = 0.0
    }
  },
  data_loader: {  # allenNLP v0.9: iterator
    batch_sampler: {
      type: 'bucket',  # try to put instances with similar length in same batch
      batch_size: 48
      # "padding_noise": 0.0 ??
    }
  },
  trainer: {
    num_epochs: epochs,
    patience: patience,  # used for early stopping
    validation_metric: '-loss',  # (default:-loss) used for early stopping
    # or "+accuracy" ?
    cuda_device: cudadevice,
    # grad_clipping: 5.0,  # todo: do we need gradient clipping?
    optimizer: {  # just using some default here
      type: 'adam',
      #lr: 0.001
    }
  }
}