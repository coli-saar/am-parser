# Copyright 2017 Jan Buys.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""Computes EDM F1 scores. Beware: python2.7"""

import sys

def strip_span_ends(triple):
  ts = triple.split(' ')
  ts[0] = ts[0].split(':')[0]
  if len(ts) >= 3 and ':' in ts[2]:
    ts[2] = ts[2].split(':')[0]
  return ' '.join(ts)

def list_predicate_spans(triples):
  spans = []
  for triple in triples:
    if len(triple.split(' ')) > 1 and triple.split(' ')[1] == 'NAME':
      spans.append(triple.split(' ')[0])  
  return spans

def inc_end_spans(spans):
  new_spans = [span.split(':')[0] + ':' + str(int(span.split(':')[1])+1)
               for span in spans]
  return new_spans

def inc_start_spans(spans):
  new_spans = [str(int(span.split(':')[0]) + 1) + ':' + span.split(':')[1]
               for span in spans]
  return new_spans

def dec_end_spans(spans):
  new_spans = [span.split(':')[0] + ':' + str(int(span.split(':')[1])-1)
               for span in spans]
  return new_spans

def compute_f1(gold_set, predicted_set, inref, verbose=False, 
    span_starts_only=False, exclude_nones=False, no_cargs=False, 
    no_predicates=False, only_predicates=False, predicates_no_span=False):
  total_gold = 0.0
  total_predicted = 0.0
  total_correct = 0.0
  none_count = 0.0

  for k, line1 in enumerate(gold_set):
    line2 = predicted_set[k]
    triples1 = [t.strip() for t in line1.split(';')]
    triples2 = [] if line2.strip() == 'NONE' else [t.strip() for t in line2.split(';')]
    if triples2 == []:
      none_count += 1
    if inref is not None and inref[k].strip() == 'NONE':
      triples1 = []
      triples2 = []

    gold_spans = set(list_predicate_spans(triples1))
    predicted_spans = list_predicate_spans(triples2)

    def replace_new_spans(new_spans):
      for i, new_span in enumerate(new_spans):
        old_span = predicted_spans[i]
        if old_span not in gold_spans and new_span in gold_spans:
          for j, triple in enumerate(triples2):
            triples2[j] = triple.replace(old_span, new_span)

    replace_new_spans(inc_end_spans(predicted_spans))
    replace_new_spans(dec_end_spans(predicted_spans))

    if span_starts_only:
      triples1 = [strip_span_ends(t) for t in triples1]

    if no_cargs:
      triples1 = filter(lambda x: (x.split(' ')[1] <> 'CARG'  
                                   if len(x.split(' ')) > 2 else True), 
                        triples1)

    if only_predicates:
      triples1 = filter(lambda x: (x.split(' ')[1] == 'NAME' 
          or x.split(' ')[1] == 'CARG' if len(x.split(' ')) > 2 else False), 
          triples1)
      if predicates_no_span:
        triples1 = map(lambda x: x.split(' ')[2], triples1) 
    elif no_predicates:
      triples1 = filter(lambda x: (x.split(' ')[1] <> 'NAME' 
          and x.split(' ')[1] <> 'CARG' if len(x.split(' ')) > 2 else False), 
          triples1)
    triples1 = set(triples1)
    if line2.strip() == 'NONE':
      if exclude_nones:
        triples1 = set()
      triples2 = set()
    else:
      if span_starts_only:
        triples2 = [strip_span_ends(t) for t in triples2]

      if no_cargs:
        triples2 = filter(lambda x: (x.split(' ')[1] <> 'CARG'  
                                     if len(x.split(' ')) > 2 else True), 
                          triples2)

      if only_predicates:
        triples2 = filter(lambda x: (x.split(' ')[1] == 'NAME' 
          or x.split(' ')[1] == 'CARG' if len(x.split(' ')) > 2 else False), 
          triples2)
        if predicates_no_span:
          triples2 = map(lambda x: x.split(' ')[2], triples2) 
      elif no_predicates:
        triples2 = filter(lambda x: (x.split(' ')[1] <> 'NAME' 
            and x.split(' ')[1] <> 'CARG' if len(x.split(' ')) > 2 else False), 
            triples2)
      triples2 = set(triples2)

    correct_triples = triples1.intersection(triples2)
    incorrect_predicted = triples2 - correct_triples
    missed_predicted = triples1 - correct_triples

    total_gold += len(triples1)
    total_predicted += len(triples2)
    total_correct += len(correct_triples)

  if total_predicted == 0 or total_gold == 0:
    print "F1: 0.0"
    return
  precision = total_correct/total_predicted
  recall = total_correct/total_gold
  f1 = 2*precision*recall/(precision+recall)

  if verbose:
    print 'Precision', precision
    print 'Recall', recall
  print 'F1-score: %.2f ' % (f1*100)
 

if __name__=='__main__':
  assert len(sys.argv) >= 3
  # Assumes in2 may contain NONE.
  in1 = open(sys.argv[1], 'r').read().split('\n') # Gold
  in2 = open(sys.argv[2], 'r').read().split('\n') # Predicted
  if len(sys.argv) >= 4 and not sys.argv[3].startswith('-'):
    inref = open(sys.argv[3], 'r').read().split('\n') # Reference
  else:
    inref = None
  no_cargs = len(sys.argv) >= 4 and sys.argv[3] == "-nocarg"
  verbose = len(sys.argv) >= 4 and sys.argv[3] == "-verbose"

  print 'All'
  compute_f1(in1, in2, inref, verbose)

  print 'All, start spans only'
  compute_f1(in1, in2, inref, verbose, span_starts_only=True)

  print 'All, predicates only'
  compute_f1(in1, in2, inref, verbose, only_predicates=True)

  print 'All, predicates only, start spans only'
  compute_f1(in1, in2, inref, verbose, span_starts_only=True, only_predicates=True)

  print 'All, predicates only - without spans'
  compute_f1(in1, in2, inref, verbose, only_predicates=True, predicates_no_span=True)

  print 'All, relations only'
  compute_f1(in1, in2, inref, verbose, no_predicates=True)

  print 'All, relations only, start spans only'
  compute_f1(in1, in2, inref, verbose, span_starts_only=True, no_predicates=True)

