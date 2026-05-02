import numpy as np

def cosine_similarity(vec1, vec2):
    v1, v2 = np.array(vec1), np.array(vec2)
    return np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))


def is_match(embedding, known_embeddings, threshold=0.4, debug=False):
    best_match = None
    best_score = -1

    for student_id, data in known_embeddings.items():
        stored_embedding = data["embedding"]

        # safety check
        if len(embedding) != len(stored_embedding):
            continue

        sim = cosine_similarity(embedding, stored_embedding)

        if sim > best_score:
            best_score = sim
            best_match = student_id

    # 🔍 Only log final result (clean)
    if debug:
        print(f"[MATCH RESULT] best: {best_match}, score: {best_score:.4f}")

    if best_score >= threshold:
        return best_match, best_score

    if debug:
        print(f"[REJECTED] score {best_score:.4f} < threshold {threshold}")

    return None, None

