import json
import re
import os

import matplotlib.pyplot as plt
import matplotlib.patches as patches
import networkx as nx
from tqdm import tqdm
import numpy as np
from noise_distribution import noise_cdfs

def explore_draw_rectangle():
    fig, ax = plt.subplots()
    ax.set_xlim(0, 200)
    ax.set_ylim(0, 200)
    add_named_rectangle(ax, 10, 100, 10, 100, "hi", "r")
    plt.show()

def add_named_rectangle(ax, x, y, width, height, label, color, facecolor):
    ax.add_patch(patches.Rectangle((x, y), width, height, linewidth=1, edgecolor=color, facecolor=facecolor))
    cx = x + width / 2
    cy = y + height / 2
    ax.text(cx, cy, label, horizontalalignment="center")

def read_biomes(filename):
    with open(filename) as f:
        biomes = json.load(f)["generator"]["biome_source"]["biomes"]
    return biomes

def generate_color_mapping():
    df_colors = pd.read_csv("biome_colors.csv", sep="\t")
    df_ids = pd.read_csv("")

def plot_biomes(biomes, first_parameter, second_parameter):
    add_args = []
    regex = re.compile("minecraft:(.*)")
    for biome in biomes:
        x = biome["parameters"][first_parameter][0]
        y = biome["parameters"][second_parameter][0]
        width = biome["parameters"][first_parameter][1] - x
        height = biome["parameters"][second_parameter][1] - y
        label = regex.match(biome["biome"]).group(1)
        add_args.append({
            "x": x,
            "y": y,
            "width": width,
            "height": height,
            "label": "",
            "color": "r",
            "facecolor": "none"
        })
    min_x = min(args["x"] for args in add_args)
    max_x = max(args["x"] + args["width"] for args in add_args)
    min_y = min(args["y"] for args in add_args)
    max_y = max(args["y"] + args["height"] for args in add_args)
    fig, ax = plt.subplots()
    dx = 0.1 * (max_x - min_x)
    dy = 0.1 * (max_y - min_y)
    ax.set_xlim(min_x - dx, max_x + dx)
    ax.set_ylim(min_y - dy, max_y + dy)
    for args in add_args:
        add_named_rectangle(ax, **args)
    plt.show()

def get_parameters(biomes):
    return biomes[0]["parameters"].keys()

def values_of_parameters(biomes):
    parameters = get_parameters(biomes)
    res = {}
    for parameter in parameters:
        # depth is not a noise and offset is always 0
        if parameter in {"depth", "offset"}:
            continue
        res[parameter] = set()
        for biome in biomes:
            if isinstance(biome["parameters"][parameter], float):
                #print(parameter)
                #return
                continue
            res[parameter].update(biome["parameters"][parameter])
        res[parameter] = list(sorted(list(res[parameter])))
    return res

def adjust_val(parameter, val):
    if parameter == "continentalness":
        if val == -1.2:
            return -10.0
    else:
        if val == -1.0:
            return -10.0
    if val == +1.0:
        return +10.0
    return val

def probabilities(biomes):
    biomes = [biome for biome in biomes if biome["parameters"]["depth"] == 0.0]
    parameters = set(get_parameters(biomes)).difference(["depth", "offset"])
    probabilities = {}
    for biome in tqdm(biomes):
        if biome["biome"] not in probabilities:
            probabilities[biome["biome"]] = 0.0
        product = 1.0
        for parameter in parameters:
            product *= proba_parameter(parameter, biome["parameters"][parameter])
        probabilities[biome["biome"]] += product
    probabilities = {biome: probabilities[biome] for biome in sorted(probabilities.keys())}
    return probabilities

def save_probabilities():
    biomes = read_biomes("overworld.json")
    with open("probabilities.json", mode="w") as f:
        json.dump(probabilities(biomes), f, indent=4)

def proba_parameter(parameter, interval):
    to_val = noise_cdfs[parameter](adjust_val(parameter, interval[1]))
    from_val = noise_cdfs[parameter](adjust_val(parameter, interval[0]))
    return to_val - from_val

def graph(biomes):
    nodes = set([biome["biome"] for biome in biomes if biome["parameters"]["depth"] == 0.0])
    parameters = set(get_parameters(biomes)).difference(["depth", "offset"])
    G = nx.Graph()
    for node in nodes:
        G.add_node(node, description=node, weight=0)
    for biome1 in tqdm(biomes):
        if biome1["parameters"]["depth"] != 0.0:
            continue
        biome1_parameters = {key: val for key, val in biome1["parameters"].items() if key in parameters}
        G.nodes[biome1["biome"]]["weight"] += np.prod(
            [
                proba_parameter(parameter, interval)
                for parameter, interval in biome1_parameters.items()
            ]
        )
        for biome2 in biomes:
            if biome2["parameters"]["depth"] != 0.0:
                continue
            if biome1["biome"] == biome2["biome"]:
                continue
            nr_adjacent = 0
            product = 1.0
            for parameter in parameters:
                length = len(set(biome1["parameters"][parameter]).intersection(biome2["parameters"][parameter]))
                from_overlap = max(biome1["parameters"][parameter][0], biome2["parameters"][parameter][0])
                to_overlap = min(biome1["parameters"][parameter][1], biome2["parameters"][parameter][1])
                if from_overlap == to_overlap:
                    nr_adjacent += 1
                    continue
                if from_overlap > to_overlap:
                    break
                product *= proba_parameter(parameter, [from_overlap, to_overlap])
            else:
                if nr_adjacent == 0:
                    print(biome1)
                    print(biome2)
                    raise RuntimeError()
                    G.add_edge(biome1["biome"], biome2["biome"])
                    continue
                # if two coordinates are adjacent, the propability for the transision is a lot less likely
                if nr_adjacent == 1:
                    pair = (biome1["biome"], biome2["biome"])
                    if G.has_edge(*pair):
                        G.edges[pair]["weight"] += product
                    else:
                        G.add_edge(*pair, weight=product)
    return G

def generate_graph(from_filename, to_filename):
    G = graph(read_biomes(from_filename))
    nx.write_graphml(G, to_filename)

def relative_graph(from_filename, to_filename):
    G = nx.read_graphml(from_filename)
    for biome1, biome2 in G.edges:
        G.edges[(biome1, biome2)]["weight"] /= np.sqrt(G.nodes[biome1]["weight"] * G.nodes[biome2]["weight"])
    for node in G.nodes:
        del G.nodes[node]["weight"]
    nx.write_graphml(G, to_filename)

def prune_graph(quantile, from_filename, to_filename):
    G = nx.read_graphml(from_filename)
    weight_arr = [G.edges[edge]["weight"] for edge in G.edges]
    split = np.quantile(weight_arr, 1 - quantile)
    for edge in G.edges:
        if G.edges[edge]["weight"] < split:
            G.remove_edge(*edge)
    nx.write_graphml(G, to_filename)

# from https://stackoverflow.com/a/65326680/3289974
def precision_round(number, digits=3):
    power = F"{number:e}".split('e')[1]
    return round(number, -(int(power) - digits))

def round_graph(from_filename, to_filename):
    G = nx.read_graphml(from_filename)
    for edge in G.edges:
        G.edges[edge]["weight"] = precision_round(G.edges[edge]["weight"], 2)
    nx.write_graphml(G, to_filename)

def show_graph():
    G = nx.read_graphml("overworld.xml")
    nx.draw(G, with_labels=True)
    plt.show()

def get_summary(biomes, first_parameter, second_parameter):
    rect_dict = {}
    for biome in biomes:
        first_interval = biome["parameters"][first_parameter]
        second_interval = biome["parameters"][second_parameter]
        rect = (*first_interval, *second_interval)
        if rect not in rect_dict:
            rect_dict[rect] = set()
        rect_dict[rect].add(biome["biome"])
    return rect_dict

def plot_summary(summary, labels, marked_biomes=None):
    min_x = min(rect[0] for rect in summary.keys())
    max_x = max(rect[1] for rect in summary.keys())
    min_y = min(rect[2] for rect in summary.keys())
    max_y = max(rect[3] for rect in summary.keys())
    d = 0.1
    dx = d * (max_x - min_x)
    dy = d * (max_y - min_y)
    fig, ax = plt.subplots()
    ax.set_xlim(min_x - dx, max_x + dx)
    ax.set_ylim(min_y - dy, max_y + dy)
    for rect in summary.keys():
        x = rect[0]
        y = rect[2]
        width = rect[1] - rect[0]
        height = rect[3] - rect[2]
        label = str(labels[rect])
        facecolor = "none"
        if marked_biomes:
            common_biomes = summary[rect].intersection(marked_biomes.keys())
            if common_biomes:
                colors = np.array([marked_biomes[biome] for biome in common_biomes])
                facecolor = np.mean(colors, axis=0)
        args = {
            "x": x,
            "y": y,
            "width": width,
            "height": height,
            "label": label,
            "color": "r",
            "facecolor": facecolor
        }
        add_named_rectangle(ax, **args)
    plt.show()

def strict_subset(rect1, rect2):
    if rect1 == rect2:
        return False
    if rect1[0] < rect2[0]:
        return False
    if rect1[1] > rect2[1]:
        return False
    if rect1[2] < rect2[2]:
        return False
    if rect1[3] > rect2[3]:
        return False
    return True

def subset(rect1, rect2):
    if rect1 == rect2:
        return True
    return strict_subset(rect1, rect2)

def find_subset_minimum(rectangles):
    element_rects = set()
    for rect1 in rectangles:
        for rect2 in rectangles:
            if strict_subset(rect2, rect1):
                break
        else:
            element_rects.add(rect1)
    return element_rects

def is_overlapping(rect1, rect2):
    if rect1 == rect2:
        return {rect1}
    if rect1[0] >= rect2[1]:
        return None
    if rect1[1] <= rect2[0]:
        return None
    if rect1[2] >= rect2[3]:
        return None
    if rect1[3] <= rect2[2]:
        return None
    res = {(
            max(rect1[0], rect2[0]),
            min(rect1[1], rect2[1]),
            max(rect1[2], rect2[2]),
            min(rect1[3], rect2[3])
        )
    }
    x_1m2 = (rect1[0], rect1[1])
    x_2m1 = (rect2[0], rect2[1])
    y_1m2 = (rect1[2], rect1[3])
    y_2m1 = (rect2[2], rect2[3])
    if rect1[0] == rect2[0]:
        x_1m2 = (rect2[1], rect1[1])
        x_2m1 = (rect1[1], rect2[1])
    if rect1[1] == rect2[1] and rect1[0] != rect2[0]:
        x_1m2 = (rect1[0], rect2[0])
        x_2m1 = (rect2[0], rect1[0])
    if rect1[2] == rect2[2] and rect1[3] != rect2[3]:
        y_1m2 = (rect2[3], rect1[3])
        y_2m1 = (rect1[3], rect2[3])
    if rect1[3] == rect2[3] and rect1[2] != rect2[2]:
        y_1m2 = (rect1[2], rect2[2])
        y_2m1 = (rect2[2], rect1[2])
    if x_1m2[1] <= x_1m2[0]:
        x_1m2 = (rect1[0], rect1[1])
    if x_2m1[1] <= x_2m1[0]:
        x_2m1 = (rect2[0], rect2[1])
    if y_1m2[1] <= y_1m2[0]:
        y_1m2 = (rect1[2], rect1[3])
    if y_2m1[1] <= y_2m1[0]:
        y_2m1 = (rect2[2], rect2[3])
    if x_1m2 and y_1m2:
        res.add((*x_1m2, *y_1m2))
    if x_2m1 and y_2m1:
        res.add((*x_2m1, *y_2m1))
    return res

def split_overlapping(rectangles):
    rectangles = list(rectangles)
    element_rects = set()
    for i, rect1 in enumerate(rectangles):
        for rect2 in rectangles[i:]:
            overlapping = is_overlapping(rect1, rect2)
            if overlapping:
                element_rects.update(overlapping)
    return element_rects

def merge_same_sets(summary):
    for rect1 in summary.keys():
        for rect2 in summary.keys():
            if rect1 == rect2:
                continue
            if summary[rect1] == summary[rect2]:
                if rect1[0] == rect2[0] and rect1[1] == rect2[1]:
                    if rect1[2] == rect2[3]:
                        res_rect = (rect1[0], rect1[1], rect2[2], rect1[3])
                    elif rect2[2] == rect1[3]:
                        res_rect = (rect2[0], rect2[1], rect1[2], rect2[3])
                    else:
                        continue
                elif rect1[2] == rect2[2] and rect1[3] == rect2[3]:
                    if rect1[0] == rect2[1]:
                        res_rect = (rect2[0], rect1[1], rect1[2], rect1[3])
                    elif rect2[0] == rect1[1]:
                        res_rect = (rect1[0], rect2[1], rect2[2], rect2[3])
                    else:
                        continue
                else:
                    continue
                #print(rect1, rect2, res_rect)
                break
        else:
            continue
        break
    else:
        return summary
    biomes = summary[rect1]
    del summary[rect1]
    del summary[rect2]
    summary[res_rect] = biomes
    return merge_same_sets(summary)

def refine_summary(summary):
    element_rects = find_subset_minimum(summary.keys())
    element_rects = split_overlapping(element_rects)
    element_rects = find_subset_minimum(element_rects)
    refined_summary = {rect: set() for rect in element_rects}
    for rect_s in summary:
        for rect_e in element_rects:
            if subset(rect_e, rect_s):
                refined_summary[rect_e].update(summary[rect_s])
    refined_summary = merge_same_sets(refined_summary)
    return refined_summary

def temporary_labeling(summary):
    labels = {}
    rectangles = list(summary.keys())
    rectangles.sort(key=lambda rect: rect[0] + rect[1])
    rectangles.sort(key=lambda rect: rect[2] + rect[3], reverse=True)
    for i, rect in enumerate(rectangles):
        labels[rect] = str(i)
    return labels

def print_summary(summary, labels):
    purged_summary = [{labels[rect]: list(biomes)} for rect, biomes in summary.items()]
    purged_summary.sort(key=lambda elem: int(next(iter(elem.keys()))))
    print(json.dumps(purged_summary, indent=4))

def show_summary(summary, marked_biomes=None):
    labels = temporary_labeling(summary)
    print_summary(summary, labels)
    plot_summary(summary, labels, marked_biomes)

def analyse_summary(filename, parameter1, parameter2, marked_biomes=None):
    biomes = read_biomes(filename)
    summary = get_summary(read_biomes("overworld.json"), parameter1, parameter2)
    summary = refine_summary(summary)
    show_summary(summary, marked_biomes)

parameters = {
    "c": "continentalness",
    "d": "depth",
    "e": "erosion",
    "h": "humidity",
    "o": "offset",
    "t": "temperature",
    "w": "weirdness"
}

def plot_main():
    marked_biomes = {
        "minecraft:desert": (1.0, 1.0, 0.0),
        "minecraft:mangrove_swamp": (0.0, 0.0, 1.0),
        #"minecraft:warm_ocean": (0.0, 1.0, 1.0)
    }
    analyse_summary("overworld.json", parameters["t"], parameters["h"], marked_biomes)    

def process_graph(from_filename, temp_filename, to_filename, quantile):
    if os.path.exists(temp_filename):
        os.remove(temp_filename)
    relative_graph(from_filename, to_filename)
    os.rename(to_filename, temp_filename)
    prune_graph(quantile, temp_filename, to_filename)
    os.remove(temp_filename)
    os.rename(to_filename, temp_filename)
    round_graph(temp_filename, to_filename)
    os.remove(temp_filename)

def main_analyse():
    save_probabilities()
    generate_graph("overworld.json", "overworld_graph.graphml")
    process_graph("overworld_graph.graphml", "temp.graphml", "overworld_graph_result_0.20_quantile.graphml", 0.20)
    process_graph("overworld_graph.graphml", "temp.graphml", "overworld_graph_result_0.35_quantile.graphml", 0.35)
    process_graph("overworld_graph.graphml", "temp.graphml", "overworld_graph_result_0.50_quantile.graphml", 0.50)
    process_graph("overworld_graph.graphml", "temp.graphml", "overworld_graph_result_full.graphml", 1.00)

if __name__ == "__main__":
    main_analyse()
   # round_graph()
   # prune_graph(0.35)
   # generate_graph("overworld.json", "overworld_graph.graphml")
   # relative_graph()
   # plot_main()
   # plot_main()
   # save_probabilities()
   # print(json.dumps(values_of_parameters(read_biomes("overworld.json")), indent=4))