import random
import json
from typing import List, Dict


def rand_step_spacing():
    return random.choice([50, 100, 150])


def generate_increasing_interconnect_stack() -> List[str]:
    tiers = [1, 2, 4, 6, 8]
    start_idx = random.randint(0, 2)
    max_depth = random.randint(2, 4)
    selected = tiers[start_idx:]

    stack = []
    last = -1
    for _ in range(max_depth):
        candidates = [x for x in selected if x > last]
        if not candidates:
            break
        next_tier = random.choice(candidates)
        stack.append(f"{next_tier}x")
        last = next_tier
    return stack


def generate_flow_with_metal_stack(prefix: str, num_device_litho_layers: int = 10, with_metal: bool = True) -> List[Dict]:
    flow = []
    step_counter = 100
    litho_block_index = 0  # Used for block prefix in step_name

    # === Layer 0.0: Pre-litho ===
    pre_litho_ops = [
        ("preclean", "R_CLN", ["CLN100", "CLN150"]),
        ("diffusion", "R_DIF", ["DIF100", "DIF110"]),
        ("poly_dep", "R_POLY", ["POLY100", "POLY110"]),
        ("implant", "R_IMP", ["IMP100", "IMP200"]),
        ("mto_poly", "R_MET", ["MET300", "MET301"]),
        ("sin_poly", "R_SIN", ["SIN100", "SIN120"]),
        ("anneal", "R_ANN", ["ANN100", "ANN120"])
    ]
    for op_name, recipe, eqp_pool in pre_litho_ops:
        numeric_base = 0
        step_name = f"{prefix}{numeric_base + step_counter:06d}"
        flow.append({
            "step_name": step_name,
            "recipe": recipe,
            "eqp_model": random.choice(eqp_pool),
            "step_type": f"pre_{op_name}",
            "layer": 0.0
        })
        step_counter += rand_step_spacing()

    # === Device Litho Layers (1.0 to N.0) ===
    for _ in range(num_device_litho_layers):
        litho_block_index += 1
        numeric_base = litho_block_index * 1000
        current_layer_id = float(litho_block_index)
        step_counter = 100

        flow.append({
            "step_name": f"{prefix}{numeric_base + step_counter:06d}",
            "recipe": "R_LTH",
            "eqp_model": random.choice(["LTH300", "LTH301"]),
            "step_type": "litho",
            "layer": current_layer_id
        })
        step_counter += rand_step_spacing()

        flow.append({
            "step_name": f"{prefix}{numeric_base + step_counter:06d}",
            "recipe": "R_ETC",
            "eqp_model": random.choice(["ETC400", "ETC401"]),
            "step_type": "etch",
            "layer": current_layer_id
        })
        step_counter += rand_step_spacing()

        flow.append({
            "step_name": f"{prefix}{numeric_base + step_counter:06d}",
            "recipe": "R_CLN",
            "eqp_model": random.choice(["CLN100", "CLN200"]),
            "step_type": "post_etch_clean",
            "layer": current_layer_id
        })
        step_counter += rand_step_spacing()

        conductor_ops = [
            ("diffusion", "R_DIF", ["DIF100", "DIF110"]),
            ("poly_dep", "R_POLY", ["POLY100", "POLY110"]),
            ("metal_dep", "R_MET", ["MET700", "MET710"])
        ]
        op, recipe, eqps = random.choice(conductor_ops)
        flow.append({
            "step_name": f"{prefix}{numeric_base + step_counter:06d}",
            "recipe": recipe,
            "eqp_model": random.choice(eqps),
            "step_type": op,
            "layer": current_layer_id
        })
        step_counter += rand_step_spacing()

        insulator_ops = [
            ("cvd", "R_CVD", ["CVD500", "CVD501"]),
            ("spin_on", "R_SOG", ["SOG100", "SOG110"])
        ]
        op, recipe, eqps = random.choice(insulator_ops)
        flow.append({
            "step_name": f"{prefix}{numeric_base + step_counter:06d}",
            "recipe": recipe,
            "eqp_model": random.choice(eqps),
            "step_type": op,
            "layer": current_layer_id
        })
        step_counter += rand_step_spacing()

    # === Interconnects ===
    if with_metal:
        interconnect_stack = generate_increasing_interconnect_stack()
        for metal_type in interconnect_stack:
            litho_block_index += 1
            numeric_base = litho_block_index * 1000
            current_layer_id = float(litho_block_index)
            step_counter = 100

            metal_structure = [
                ("litho", "R_LTH", ["LTH300", "LTH301"]),
                ("etch", "R_ETC", ["ETC400", "ETC401"]),
                ("metal_dep", "R_MET", ["MET700", "MET710"]),
                ("cmp", "R_CMP", ["CMP800", "CMP810"])
            ]

            for op_name, recipe, eqp_pool in metal_structure:
                step_name = f"{prefix}{numeric_base + step_counter:06d}"
                flow.append({
                    "step_name": step_name,
                    "recipe": recipe,
                    "eqp_model": random.choice(eqp_pool),
                    "step_type": f"{metal_type}_{op_name}",
                    "layer": current_layer_id
                })
                step_counter += rand_step_spacing()

                if op_name == "etch":
                    clean_step_name = f"{prefix}{numeric_base + step_counter:06d}"
                    flow.append({
                        "step_name": clean_step_name,
                        "recipe": "R_CLN",
                        "eqp_model": random.choice(["CLN200", "CLN250"]),
                        "step_type": f"{metal_type}_post_etch_clean",
                        "layer": current_layer_id
                    })
                    step_counter += rand_step_spacing()

                if op_name == "cmp":
                    for _ in range(random.choice([1, 2])):
                        ild_step_name = f"{prefix}{numeric_base + step_counter:06d}"
                        flow.append({
                            "step_name": ild_step_name,
                            "recipe": "R_ILD",
                            "eqp_model": random.choice(["CVD900", "CVD950"]),
                            "step_type": f"{metal_type}_inter_layer_dielectric",
                            "layer": current_layer_id
                        })
                        step_counter += rand_step_spacing()

    return flow


def generate_multiple_flows(out_file="data/example_flows.json"):
    flows = {
        "FX": generate_flow_with_metal_stack("FX", num_device_litho_layers=10, with_metal=True),
        "LX": generate_flow_with_metal_stack("LX", num_device_litho_layers=10, with_metal=True),
        "TX": generate_flow_with_metal_stack("TX", num_device_litho_layers=10, with_metal=False)
    }
    with open(out_file, "w") as f:
        json.dump(flows, f, indent=2)
    print(f"✅ Saved example flows to {out_file}")


if __name__ == "__main__":
    generate_multiple_flows()
