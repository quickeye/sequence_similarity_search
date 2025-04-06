# sequence_similarity_search/utils/flow_utils.py

def chunk_flow_by_layer_id(flow):
    chunks = {}
    
    def get_layer_id(step_name):
        prefix = step_name[2:5]
        return float(int(prefix)) / 10.0

    for step in flow:
        layer_id = get_layer_id(step["step_name"])
        if layer_id not in chunks:
            chunks[layer_id] = []
        chunks[layer_id].append(step)

    return [(lid, chunks[lid]) for lid in sorted(chunks.keys())]
