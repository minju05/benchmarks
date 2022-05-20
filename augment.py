import os
import json
import random

def random_deletion(words, p):
    # 단어 하나만 등장할 경우 delete 안함
    if len(words) == 1:
        return words

    new_words = []
    for word in words:
        # 랜덤한 비율을 추출
        r = random.uniform(0, 1)
        # 랜덤한 비율이 delete 비율인 p보다 클 경우만 해당 단어 사용, 아닌 경우 랜덤 삭제
        if r > p:
            new_words.append(word)

    # 아무것도 단어가 남지 않는 경우 무작위로 index 하나 뽑아서 한 단어만 남기기
    if len(new_words) == 0:
        rand_int = random.randint(0, len(words) - 1)
        return [words[rand_int]]

    return new_words


def random_swap(words, n):
    new_words = words.copy()
    for _ in range(n):
        new_words = swap_word(new_words)
    return new_words


def swap_word(words):
    # 바꿀 인덱스 두개 랜덤하게 추출
    random_idx_1 = random.randint(0, len(words) - 1)
    random_idx_2 = random_idx_1

    # 두번째 바꿀 인덱스 랜덤하게 추출
    counter = 0
    while random_idx_2 == random_idx_1:
        random_idx_2 = random.randint(0, len(words) - 1)
        counter += 1
        # 세번 랜덤하게 해도 동일한 인덱스 추출하면 그냥 return
        if counter > 3:
            return words
    # 두 인덱스에 대해서 단어 swap
    words[random_idx_1], words[random_idx_2] = words[random_idx_2], words[random_idx_1]
    return words


def EDA(df, alpha_sr=0.1, alpha_ri=0.1, alpha_rs=0.1, p_rd=0.1, num_aug=9):
    sentence = df['comments']
    # 문장을 단어로 split
    words = sentence.split(' ')
    words = [word for word in words if word != ""]
    num_words = len(words)

    augmented_sentences = []
    # 몇번이나 augmentation 진행할지 -> 임의로 전체 단어 길이/4+1
    num_new_per_technique = int(num_aug / 4) + 1

    n_rs = max(1, int(alpha_rs * num_words))

    # random swap 결과
    for _ in range(num_new_per_technique):
        a_words = random_swap(words, n_rs)
        augmented_sentences.append(" ".join(a_words))

    # random delete 결과
    for _ in range(num_new_per_technique):
        a_words = random_deletion(words, p_rd)
        augmented_sentences.append(" ".join(a_words))

        # 데이터셋 담기
        augmented_sentences = [sentence for sentence in augmented_sentences]
        random.shuffle(augmented_sentences)

        if num_aug >= 1:
            augmented_sentences = augmented_sentences[:num_aug]
        else:
            keep_prob = num_aug / len(augmented_sentences)
            augmented_sentences = [s for s in augmented_sentences if random.uniform(0, 1) < keep_prob]

        augmented_sentences.append(sentence)
        return augmented_sentences