# config file for full laptop-based data set (while implementing/debugging)
# allennlp train -f --include-package typetagging -s ./typetagging/tmp/full ./typetagging/fullconfig.jsonnet
# tensorboard --logdir ./typetagging/tmp/full
# allennlp evaluate --include-package typetagging ./typetagging/tmp/full/model.tar.gz ~/HiwiAK/data/sempardata/ACL2019/SemEval/2015/@@SPLITMARKER@@/gold-dev/gold-dev.amconll
#
# tested with allennlp 0.8.4 (same version as mentioned in readme)
# note:
# todo: add more dropout maybe? (input, encoder, classifier ...)

local epochs = 50;
local patience = 2;
local cudadevice = 0;  # -1

# local maxtrainsize = null;  # full dataset
local maxtrainsize = 1000;  # todo currently deterministic for debugging

# local source_target_pair = ['PAS', 'DM-low-100-3'];
local source_target_pair = ['PAS', 'DM'];
# local source_target_pair = ['PAS', 'PSD'];

# in train/valid path and dataset_reader init parameter:
# insertion point for source_target_pair
local splitmarker = '@@SPLITMARKER@@';
local common_prefix = '/home/wurzel/HiwiAK/data/sempardata/ACL2019/SemEval/2015/';


local encoder_hidden_size = 128;
local input_dropout = 0.2;  # todo if non-zero adjust elmo dropout
local encoder_dropout = 0.3;

# todo: is there a more convenient way?
local include_pos = false;
local include_srctypes = true;

local pos_dim = if include_pos then 32 else 0;
local src_type_dim = if include_srctypes then 32 else 0;
local pos_emb = {embedding_dim: pos_dim, vocab_namespace: "src_pos"};
local src_type_emb = {embedding_dim: src_type_dim, vocab_namespace: "src_types"};


## Word embedding options: --------------------------------------------------- #
local embedding_name = "elmo";  # <--- change to one of the options (bert, elmo, glove, token)
local finetune_pretrained_embed = false;  # todo test finetuning

# (1) glove
local glove_dim = 200;
local glove_file = "/home/wurzel/HiwiAK/data/pretrained-embeddings/glove/"+"glove.6B.200d.txt";
# local glove_file = "(http://nlp.stanford.edu/data/glove.6B.zip)#glove.6B.200d.txt"
# (2) bert
local bert_model = "bert-base-uncased";
local bert_dim = 768;  # bert-base-uncased: 768
# (3) elmo
local elmo_dim = 128*2;
# https://s3-us-west-2.amazonaws.com/allennlp/models/elmo/2x1024_128_2048cnn_1xhighway/elmo_2x1024_128_2048cnn_1xhighway_options.json
local elmo_option_file = "/home/wurzel/HiwiAK/data/pretrained-embeddings/elmo/"+"elmo_small_2x1024_128_2048cnn_1xhighway_options.json";
# https://s3-us-west-2.amazonaws.com/allennlp/models/elmo/2x1024_128_2048cnn_1xhighway/elmo_2x1024_128_2048cnn_1xhighway_weights.hdf5
local elmo_weight_file = "/home/wurzel/HiwiAK/data/pretrained-embeddings/elmo/"+"elmo_small_2x1024_128_2048cnn_1xhighway_weights.hdf5";
# (4) token   // just a trainable standard embedding
local token_dim = 200;


# (0) The word embedding switch, part 0:
#     dimension depends on wether we use GloVe, BERT, ELMo or a default implementation
local getWordEmbeddingDim(name="glove") =
    if name == "glove" then
        glove_dim
    else if name == "bert" then
        bert_dim
    else if name == "elmo" then
        elmo_dim
    else
        token_dim;


# (1) Word embedding switch, part 1: getIndexer
#     source_target_indexer in the dataset reader
local getIndexer(name='glove') =
    if name == 'glove' then
        { 'glove' :
            { type: 'single_id',
                lowercase_tokens: true,
                namespace: 'src_words'  # todo: needed?
            }
        }
    else if name == 'bert' then
        { 'bert':
            { type: 'bert-pretrained',  # : 'pretrained_transformer_mismatched'
                pretrained_model: bert_model,  # model_name: bert_model,
                # namespace: 'src_words'  # todo why is this not allowed?
                # do_lowercase: true
                # use_starting_offsets: false # if true use first instead of last piece
            }
        }
    else if name == 'elmo' then
        {'elmo': {type: 'elmo_characters', namespace: 'src_words'}}
    else
        # error "Unknown type for embedding, use 'bert' or 'glove'";
        {tokens: {type: 'single_id', namespace: 'src_words'}};


# (2) Word embedding switch, part 2: getEmbedder
#     text_field_embedder in the model
local getEmbedder(name='glove') =
    if name == 'glove' then
        { 'glove':
            { type: "embedding",
                vocab_namespace: "src_words",  # todo: needed?
                embedding_dim: glove_dim,
                pretrained_file: glove_file,
                trainable: finetune_pretrained_embed
            }
        }
    else if name == 'bert' then
        {
            allow_unmatched_keys: true,
            embedder_to_indexer_map: { 'bert': ["bert", "bert-offsets"] },
            'bert': {
                type: "bert-pretrained", # "pretrained_transformer_mismatched"
                pretrained_model: bert_model,  # model_name : bert_model,
                # vocab_namespace: "src_words",  # todo why is this not allowed?
                # train_parameters: finetune_pretrained_embed
                requires_grad: finetune_pretrained_embed
            }
        }
    else if name == 'elmo' then
        {
            'elmo': {
                type: 'elmo_token_embedder',
                options_file: elmo_option_file,
                weight_file: elmo_weight_file,
                do_layer_norm: false,
                dropout: 0, # 0.5  todo set to 0 if dropout is applied already on the input
                requires_grad: finetune_pretrained_embed,
                # vocab_namespace: "src_words",  # todo why is this not allowed?
            }
        }
    else
        # some allennlp versions require additionally wrapping below object with
        # token_embedders: { ... }
        {
            tokens: {
                type: 'embedding',
                vocab_namespace: 'src_words',
                embedding_dim: token_dim,
                trainable: true
            }
        };


## Encoder, Classifier and Model --------------------------------------------- #

# encoder input consists of word embedding and optionally src type embedding
# and pos embedding (are 0 if disabled).
# embedding dimension varies (glove vs bert vs ...)
local encoder_input_dim = getWordEmbeddingDim(embedding_name) + src_type_dim + pos_dim;

# Encoder gets words (and evtl. pos tags and src types) as input,
# the output (vector per token position) is passed on to the classifier
local encoder = {
    # type: 'stacked_bidirectional_lstm',  # -> StackedBidirectionalLstm
    type: 'lstm',  # -> LstmSeq2SeqEncoder, actually a bilstm, see below
    # todo: influence of type on whether need *2 for bidirectionality elsewhere?
    num_layers: 2,
    input_size: encoder_input_dim,
    hidden_size: encoder_hidden_size,
    # --would require: lstm   as type:
    # parameters from allennlp.modules.LstmSeq2SeqEncoder (pytorch wrapper)
    bidirectional: true,
    dropout: encoder_dropout, # default: 0
    # -- would require: stacked_bidirectional_lstm  as type:
    # parameters from allennlp.modules.StackedBiDirectionalLSTM: (slower)
    # recurrent_dropout_probability: 0.0, # float, optional (default = 0.0)
    # layer_dropout_probability: 0.0, # float, optional (default = 0.0)
    # use_highway: bool, optional (default = True)
};

# Classifier receives encoder vectors as input,
# predicts best target graph constant type per token position
local classifier = {
    # parameter names from allennlp.modules.FeedForward:
    input_dim: encoder_hidden_size * 2, # *2 for bidirectionality of lstm
    num_layers: 1,
    hidden_dims: [128],
    activations: ['relu'],
    #dropout: 0.3 # dropout: Union[float, List[float]] = 0.0
};


local model = {
    type: 'typetaggingmodel', # see corresponding py class and file
    # below follow parameters of TypeTaggingModel.__init__
    text_field_embedder: getEmbedder(embedding_name),
    # note: because pos tags and src types can be disabled as input,
    # old allennlp version doesn't allow me to set it to null directly
    # (otherwise it seems to attempt to build an Embedding object from None),
    # so I had to create a function getFinalModel (see below) that will add
    # the embeddings only if needed. # todo is there an easier way?
    ##pos_tag_embedding: pos_emb,
    ##src_type_embedding: src_type_emb,
    input_dropout: input_dropout,
    encoder: encoder,
    classifier: classifier
};

local getFinalModel() =
    if include_pos then
        if include_srctypes then
            model + { pos_tag_embedding: pos_emb, src_type_embedding: src_type_emb }
        else
            model + { pos_tag_embedding: pos_emb}
    else
        if include_srctypes then
            model + {src_type_embedding: src_type_emb }
        else
            model
;

{
  dataset_reader: {
    type: 'typeamconllreader', # see typeamconllreader.py
    lazy: false,
    source_token_indexers: getIndexer(embedding_name),
    source_target_foldername_pair: source_target_pair, # ['DM', 'PAS']
    splitmarker: splitmarker,
    maxtrainsize: maxtrainsize, # default is: null
  },
  train_data_path: common_prefix + splitmarker + '/train/train.amconll',
  validation_data_path: common_prefix + splitmarker + '/gold-dev/gold-dev.amconll',
  model: getFinalModel(),
  iterator: {
    type: 'bucket',
    sorting_keys: [['src_words', 'num_tokens']],
    batch_size: 48,
    # newer allennlp:
    # data_loader: { # instead of iterator
    # batch_sampler: {
    #  type: 'bucket',  # try to put instances with similar length in same batch
    #  batch_size: 48
    #  # "padding_noise": 0.0 ??
    #}
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