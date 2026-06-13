PAD = '[PAD]'
BOS = '[CLS]'
EOS = '[SEP]'
UNK = '[UNK]'

GRAMMAR_FILEPATH = 'asdls/sql/grammar/sql_asdl_v2.txt'
SCHEMA_TYPES = ['table', 'others', 'text', 'time', 'number', 'boolean']
MAX_RELATIVE_DIST = 2
# relations: type_1-type_2-rel_name, r represents reverse edge, b represents bidirectional edge
# RELATIONS = ['question-question-dist' + str(i) if i != 0 else 'question-question-identity' for i in range(- MAX_RELATIVE_DIST, MAX_RELATIVE_DIST + 1)] + \
#     ['table-table-identity', 'table-table-fk', 'table-table-fkr', 'table-table-fkb'] + \
#     ['column-column-identity', 'column-column-sametable', 'column-column-fk', 'column-column-fkr'] + \
#     ['table-column-pk', 'column-table-pk', 'table-column-has', 'column-table-has'] + \
#     ['question-column-exactmatch', 'question-column-partialmatch', 'question-column-nomatch', 'question-column-valuematch',
#     'column-question-exactmatch', 'column-question-partialmatch', 'column-question-nomatch', 'column-question-valuematch'] + \
#     ['question-table-exactmatch', 'question-table-partialmatch', 'question-table-nomatch',
#     'table-question-exactmatch', 'table-question-partialmatch', 'table-question-nomatch'] + \
#     ['question-question-generic', 'table-table-generic', 'column-column-generic', 'table-column-generic', 'column-table-generic'] + \
#     ['*-*-identity', '*-question-generic', 'question-*-generic', '*-table-generic', 'table-*-generic', '*-column-generic', 'column-*-generic']

RELATIONS = ['question-question-dist' + str(i) if i != 0 else 'question-question-identity' for i in range(- MAX_RELATIVE_DIST, MAX_RELATIVE_DIST + 1)] + \
    ['table-table-identity', 'table-table-fk', 'table-table-fkr', 'table-table-fkb'] + \
    ['column-column-identity', 'column-column-sametable', 'column-column-fk', 'column-column-fkr'] + \
    ['table-column-pk', 'column-table-pk', 'table-column-has', 'column-table-has'] + \
    ['question-column-exactmatch', 'question-column-partialmatch','question-column-partialsemanticmatch', 'question-column-nomatch','question-column-semanticmatch', 'question-column-valuematch',
    'column-question-exactmatch', 'column-question-partialmatch','column-question-partialsemanticmatch', 'column-question-nomatch','column-question-semanticmatch', 'column-question-valuematch'] + \
    ['question-table-exactmatch', 'question-table-partialmatch','question-table-partialsemanticmatch', 'question-table-nomatch','question-table-semanticmatch',
    'table-question-exactmatch', 'table-question-partialmatch', 'table-question-partialsemanticmatch','table-question-semanticmatch','table-question-nomatch'] + \
    ['question-question-generic', 'table-table-generic', 'column-column-generic', 'table-column-generic', 'column-table-generic'] + \
    ['*-*-identity', '*-question-generic', 'question-*-generic', '*-table-generic', 'table-*-generic', '*-column-generic', 'column-*-generic'] + \
    ['pk-pk-identity', 'pk-pk-generic', 'pk-table-of', 'table-pk-of', 'pk-table-generic', 'table-pk-generic',
     'pk-column-member', 'column-pk-member', 'pk-column-generic', 'column-pk-generic',
     'question-pk-nomatch', 'pk-question-nomatch'] + \
    ['fk-fk-identity', 'fk-fk-generic', 'fk-table-src', 'table-fk-src', 'fk-table-tgt', 'table-fk-tgt',
     'fk-table-generic', 'table-fk-generic', 'fk-column-src', 'column-fk-src', 'fk-column-tgt', 'column-fk-tgt',
     'fk-column-generic', 'column-fk-generic', 'fk-pk-generic', 'pk-fk-generic',
     'question-fk-nomatch', 'fk-question-nomatch'] + \
    ['path-path-identity', 'path-path-generic',
     'path-table-member', 'table-path-member', 'path-table-generic', 'table-path-generic',
     'path-column-member', 'column-path-member', 'path-column-generic', 'column-path-generic',
     'path-pk-generic', 'pk-path-generic', 'path-fk-member', 'fk-path-member',
     'path-fk-generic', 'fk-path-generic', 'question-path-nomatch', 'path-question-nomatch'] + \
    ['question-schema-dynamic', 'schema-question-dynamic']

RELATIONS_INDEX = {}
index = 0
for item in RELATIONS:
    RELATIONS_INDEX[item] = index
    index += 1
# import ipdb; ipdb.set_trace()
