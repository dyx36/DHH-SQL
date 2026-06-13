#coding=utf8
import os, json, pickle, argparse, sys, time
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from preprocess.graph_utils import GraphProcessor
from utils.hypergraph_metadata import build_hypergraph_metadata, validate_hypergraph_metadata
import time


def process_dataset_graph(processor, dataset, tables, output_path=None, skip_large=False):
    processed_dataset = []
    for idx, entry in enumerate(dataset):
        db = tables[entry['db_id']]
        if skip_large and len(db['column_names']) > 100:
            continue
        if (idx + 1) % 500 == 0:
            print('Processing the %d-th example ...' % (idx + 1))
        entry = processor.process_graph_utils(entry, db)
        processed_dataset.append(entry)

    print('In total, process %d samples, skip %d samples .' % (len(processed_dataset), len(dataset) - len(processed_dataset)))
    if output_path is not None:
        pickle.dump(processed_dataset, open(output_path, 'wb'))
    return processed_dataset


if __name__ == '__main__':

    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument('--dataset_path', type=str, required=True, help='dataset path')
    arg_parser.add_argument('--table_path', type=str, required=True, help='processed table path')
    arg_parser.add_argument('--output_path', type=str, required=True, help='output preprocessed dataset')
    arg_parser.add_argument('--max_fk_path_len', type=int, default=3,
                            help='expected maximum FK hops used when tables.bin was generated')
    arg_parser.add_argument('--paths_per_table_pair', type=int, default=2,
                            help='expected number of shortest join paths retained per table pair')
    args = arg_parser.parse_args()

    processor = GraphProcessor()
    # loading database and dataset
    tables = pickle.load(open(args.table_path, 'rb'))
    validate_hypergraph_metadata(
        tables,
        args.table_path,
        expected=build_hypergraph_metadata(args.max_fk_path_len, args.paths_per_table_pair)
    )
    dataset = pickle.load(open(args.dataset_path, 'rb'))
    start_time = time.time()
    dataset = process_dataset_graph(processor, dataset, tables, args.output_path)
    print('Dataset preprocessing costs %.4fs .' % (time.time() - start_time))
