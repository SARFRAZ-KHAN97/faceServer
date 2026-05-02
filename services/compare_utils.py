import numpy as np

def cosine_similarity(vec1, vec2):
    v1, v2 = np.array(vec1), np.array(vec2)
    return np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))


def is_match(embedding, known_embeddings, threshold=0.55):
    best_match = None
    best_score = -1

    for student_id, data in known_embeddings.items():
        sim = cosine_similarity(embedding, data["embedding"])
        if sim > threshold and sim > best_score:
            best_match = student_id
            best_score = sim

    if best_match is None:
        return None, None
    return best_match, best_score

