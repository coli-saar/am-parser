#/bin/bash

find data/ | grep train.amconll | sed -E "s/train\/train.amconll//" | xargs -I{} ./scripts/prepare_extra_lexicon.sh {}
