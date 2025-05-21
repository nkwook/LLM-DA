import argparse
import os

import torch
import numpy as np
import time

from grapher import Grapher
from rule_learning import Rule_Learner, rules_statistics
from temporal_walk import initialize_temporal_walk
from joblib import Parallel, delayed
from datetime import datetime

os.environ["CURL_CA_BUNDLE"] = ""
os.environ["TOKENIZERS_PARALLELISM"] = "false"

from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from params import str_to_bool

from utils import load_json_data, save_json_data


def select_similary_relations(relation2id, output_dir):
    id2relation = dict([(v, k) for k, v in relation2id.items()])

    save_json_data(id2relation, os.path.join(output_dir, "transfomers_id2rel.json"))
    save_json_data(relation2id, os.path.join(output_dir, "transfomers_rel2id.json"))

    all_rels = list(relation2id.keys())
    # 사전 훈련된 모델 로드
    model = SentenceTransformer("bert-base-nli-mean-tokens")

    # 문장 정의
    sentences_A = all_rels
    sentences_B = all_rels

    # 모델을 사용하여 문장 인코딩
    embeddings_A = model.encode(sentences_A)
    embeddings_B = model.encode(sentences_B)

    # 문장 간 코사인 유사도 계산
    similarity_matrix = cosine_similarity(embeddings_A, embeddings_B)

    np.fill_diagonal(similarity_matrix, 0)

    np.save(os.path.join(output_dir, "matrix.npy"), similarity_matrix)


def main(parsed):
    dataset = parsed["dataset"]
    rule_lengths = parsed["max_path_len"]
    rule_lengths = (torch.arange(rule_lengths) + 1).tolist()
    num_walks = parsed["num_walks"]
    transition_distr = parsed["transition_distr"]
    num_processes = parsed["cores"]
    seed = parsed["seed"]
    version_id = parsed["version"]

    dataset_dir = "./datasets/" + dataset + "/"
    data = Grapher(dataset_dir)

    temporal_walk = initialize_temporal_walk(version_id, data, transition_distr)

    rl = Rule_Learner(temporal_walk.edges, data.id2relation, data.inv_relation_id, dataset)
    all_relations = sorted(temporal_walk.edges)  # 모든 관계에 대해 학습
    all_relations = [int(item) for item in all_relations]
    rel2idx = data.relation2id

    select_similary_relations(data.relation2id, rl.output_dir)

    constant_config = load_json_data("./Config/constant.json")
    relation_regex = constant_config["relation_regex"][dataset]

    def learn_rules(i, num_relations, use_relax_time=False):
        """
        선택적 완화 시간(멀티프로세싱 가능)으로 규칙을 학습합니다.

        매개변수:
            i (int): 프로세스 번호
            num_relations (int): 각 프로세스에 대한 최소 관계 수
            use_relax_time (bool): 샘플링 시 완화 시간 사용 여부

        반환값:
            rl.rules_dict (dict): 규칙 사전
        """

        set_seed_if_provided()
        relations_idx = calculate_relations_idx(i, num_relations)
        num_rules = [0]

        for k in relations_idx:
            rel = all_relations[k]
            for length in rule_lengths:
                it_start = time.time()
                process_rules_for_relation(rel, length, use_relax_time)
                it_end = time.time()
                it_time = round(it_end - it_start, 6)
                num_rules.append(sum([len(v) for k, v in rl.rules_dict.items()]) // 2)
                num_new_rules = num_rules[-1] - num_rules[-2]

                print(
                    f"Process {i}: relation {k - relations_idx[0] + 1}/{len(relations_idx)}, length {length}: {it_time} sec, {num_new_rules} rules"
                )

        return rl.rules_dict

    def set_seed_if_provided():
        if seed:
            np.random.seed(seed)

    def calculate_relations_idx(i, num_relations):
        if i < num_processes - 1:
            return range(i * num_relations, (i + 1) * num_relations)
        else:
            return range(i * num_relations, len(all_relations))

    def process_rules_for_relation(rel, length, use_relax_time):
        for _ in range(num_walks):
            walk_successful, walk = temporal_walk.sample_walk(length + 1, rel, use_relax_time)
            if walk_successful:
                rl.create_rule(walk, use_relax_time)

    start = time.time()
    num_relations = len(all_relations) // num_processes
    output = Parallel(n_jobs=num_processes)(
        delayed(learn_rules)(i, num_relations, parsed["is_relax_time"]) for i in range(num_processes)
    )
    end = time.time()
    all_graph = output[0]
    for i in range(1, num_processes):
        all_graph.update(output[i])

    total_time = round(end - start, 6)
    print("학습 완료: {} 초".format(total_time))

    rl.rules_dict = all_graph
    rl.sort_rules_dict()
    dt = datetime.now()
    dt = dt.strftime("%d%m%y%H%M%S")
    rl.save_rules(dt, rule_lengths, num_walks, transition_distr, seed)
    save_json_data(rl.rules_dict, rl.output_dir + "confidence.json")
    rules_statistics(rl.rules_dict)
    rl.save_rules_verbalized(dt, rule_lengths, num_walks, transition_distr, seed, rel2idx, relation_regex)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, default="datasets", help="데이터 디렉토리")
    parser.add_argument("--dataset", "-d", default="", type=str)
    parser.add_argument("--max_path_len", "-m", type=int, default=3, help="최대 샘플링된 경로 길이")
    parser.add_argument("--anchor", type=int, default=5, help="각 관계에 대한 앵커 사실")
    parser.add_argument("--output_path", type=str, default="sampled_path", help="출력 경로")
    parser.add_argument("--sparsity", type=float, default=1, help="데이터셋 샘플링 희소성")
    parser.add_argument("--cores", "-p", type=int, default=5, help="사용할 코어 수")
    parser.add_argument("--num_walks", "-n", default="100", type=int)
    parser.add_argument("--transition_distr", default="exp", type=str)
    parser.add_argument("--seed", "-s", default=None, type=int)
    parser.add_argument("--window", "-w", default=0, type=int)
    parser.add_argument("--version", default="train", type=str, choices=["train", "test", "train_valid", "valid"])
    parser.add_argument("--is_relax_time", default="no", type=str_to_bool)

    parsed = vars(parser.parse_args())

    main(parsed)
