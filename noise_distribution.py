import os
import sys
import time
import json
import random
from collections.abc import Iterable

import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import scipy.stats as ss
import numpy as np
from scipy.interpolate import interp1d
from tqdm import tqdm

INPUT_FACTOR = 1.0181268882175227
TARGET_DEVIATION = 0.3333333333333333
GRADIENTS = [
    (1, 1, 0),
    (-1, 1, 0),
    (1, -1, 0),
    (-1, -1, 0),
    (1, 0, 1),
    (-1, 0, 1),
    (1, 0, -1),
    (-1, 0, -1),
    (0, 1, 1),
    (0, -1, 1),
    (0, 1, -1),
    (0, -1, -1),
    (1, 1, 0),
    (0, -1, 1),
    (-1, 1, 0),
    (0, -1, -1)
]

# this function doesn't work properly, the values don't have the correct distribution
def normal_vals(n: int, amplitudes=[1]):
    m = n ** (1 / 3)
    m = int(np.ceil(m))
    n = m * m * m
    vals = []
    for ind in range(n):
        x = (ind % m) / m / m - 0.5
        y = ((ind // m) % m) / m - 0.5
        z = ((ind // m // m) % m) / m - 0.5
        vals.append(normal_noise(x, y, z, amplitudes))
    return list(sorted(vals))

def expected_deviation(i):
    return 0.1 * (1.0 + 1.0 / (i + 1))

def edge_value(d, amplitudes):
    i = len(amplitudes)
    f = 2 ** (i - 1) / (2 ** i - 1)
    res = 0.0
    for amplitude in amplitudes:
        res += amplitude * d * f
        f /= 2.0
    return res

def value_factor(amplitudes):
    relevant_inds = [ind for ind, amplitude in enumerate(amplitudes) if amplitude != 0.0]
    j = min(relevant_inds)
    k = max(relevant_inds)
    return 0.16666666666666666 / expected_deviation(k - j)

def normal_noise(x, y, z, amplitudes, max_val=False, min_val=False):
    assert not max_val or not min_val
    this_value_factor = value_factor(amplitudes)
    max_value = (
            fractual_perlin_noise(None, None, None, amplitudes, max_val=True)
            + fractual_perlin_noise(None, None, None, amplitudes, max_val=True)
    ) * this_value_factor
    min_value = (
            fractual_perlin_noise(None, None, None, amplitudes, min_val=True)
            + fractual_perlin_noise(None, None, None, amplitudes, min_val=True)
    ) * this_value_factor
    if max_val:
        return max_value
    if min_val:
        return min_value
    args = x, y, z, amplitudes
    # minecraft also adds two perlin noises, easier here than folding
    return (fractual_perlin_noise(*args) + fractual_perlin_noise(*args)) * this_value_factor

def fractual_perlin_noise(x, y, z, amplitudes, max_val=False, min_val=False):
    assert not max_val or not min_val
    noise_val = 0
    for ind, amplitude in enumerate(amplitudes):
        if min_val or max_val:
            if max_val == (amplitude > 0):
                res = edge_value(2.0, amplitudes)
                return res
            else:
                return -edge_value(2.0, amplitudes)
        else:
            x_val = x * 2 ** ind
            x_val = x_val % 1.0
            y_val = y * 2 ** ind
            y_val = y_val % 1.0
            z_val = z * 2 ** ind
            z_val = z_val % 1.0
            noise_val = np.sqrt(2) * perlin_noise(x_val, y_val, z_val)
        noise_val += amplitude * noise_val / 2 ** ind
    return noise_val

def lerp1d(p, arr):
    return arr[0] + (arr[1] - arr[0]) * p

def lerp2d(p, q, arr):
    return lerp1d(p, [lerp1d(q, arr[0]), lerp1d(q, arr[1])])

def lerp3d(p, q, r, arr):
    return lerp1d(p, [lerp2d(q, r, arr[0]), lerp2d(q, r, arr[1])])

def dotVal(x, y, z):
    grad = random.choice(GRADIENTS)
    return grad[0] * x + grad[1] * y + grad[2] * z

def perlin_noise(x, y, z):
    arr = [[[dotVal(x, y, z) for _ in range(2)] for _ in range(2)] for _ in range(2)]
    return lerp3d(x, y, z, arr)

def generate_function(filename, n: int, amplitudes: list[int]):
    if filename and os.path.exists(filename):
        with open(filename) as f:
            vals = [float(elem) for elem in f.read().strip().split()]
    else:
        # the function is broken. It isn't worth it to
        # fix this function if I have the values directly from minecraft
        assert False
        vals = normal_vals(n, amplitudes)
        vals = list(sorted(set(vals)))
        if filename:
            with open(filename, mode="w") as f:
                f.write("\n".join(str(elem) for elem in vals))
    n = len(vals)
    room = list(np.linspace(0, 1, n))
    # linear works best
    return LinearInterpolation(vals, room, 0.0, 1.0)

# because numpy is too slow and scipy is deprecated, it's best to write it on my own
class LinearInterpolation():
    def __init__(self, x_vals, y_vals, first_val, last_val):
        assert len(x_vals) == len(y_vals)
        self.x_vals = x_vals
        self.y_vals = y_vals
        self.n = len(x_vals)
        self.first_val = first_val
        self.last_val = last_val
        self.start = x_vals[0]
        self.end = x_vals[-1]
    def __call__(self, x):
        if isinstance(x, Iterable):
            return [self(val) for val in x]
        start_ind = 0
        end_ind = self.n - 1
        mid = (start_ind + end_ind) // 2
        while start_ind + 1 < end_ind:
            if self.x_vals[mid] < x:
                start_ind = mid
            elif self.x_vals[mid] == x:
                return self.y_vals[mid]
            else:
                end_ind = mid
            mid = (start_ind + end_ind) // 2
        if start_ind == 0 and self.x_vals[0] > x:
            return self.first_val
        if end_ind == self.n - 1 and self.x_vals[-1] < x:
            return self.last_val
        assert self.x_vals[start_ind] < x
        assert self.x_vals[end_ind] > x
        p = x - self.x_vals[start_ind]
        return (1 - p) * self.y_vals[start_ind] + p * self.y_vals[start_ind + 1]

def compare_func_computationally(func1, func2, n=1_000):
    start = max(func1.start, func2.start)
    end = min(func1.end, func2.end)
    room = np.linspace(start, end, n)
    return np.sqrt(np.sum((func1(room) - func2(room)) ** 2 / n))

def compare_func_visually(func1, func2, n=1_000):
    start = max(func1.start, func2.start)
    end = min(func1.end, func2.end)
    room = np.linspace(start, end, n)
    plt.plot(room, func1(room), "r")
    plt.plot(room, func2(room), "g")
    plt.show()

def check_first_octave_not_needed():
    n = 30_000
    room_translate = list(np.linspace(0, 1, n)) + [1.0]
    func1 = generate_function(None, n, [1])
    first_octave = 5
    func2 = generate_function(None, n, first_octave * [0.0] + [2 ** first_octave])
    sample_derivation = compare_func_computationally(func1, func2, 10_000)
    assert sample_derivation <= 3e-3
    print(f"{sample_derivation=}")
    compare_func_visually(func1, func2, 10_000)

print("starting...")

noise_cdfs = {}
amplitudes = {
    "temperature": [1.5, 0.0, 1.0, 0.0, 0.0, 0.0],
    "humidity": [1.0, 1.0, 0.0, 0.0, 0.0, 0.0],
    "continentalness": [1.0, 1.0, 2.0, 2.0, 2.0, 1.0, 1.0, 1.0, 1.0],
    "erosion": [1.0, 1.0, 0.0, 1.0, 1.0],
    "weirdness": [1.0, 2.0, 1.0, 0.0, 0.0, 0.0]
}

n = 100_000
for noise, noise_amplitudes in tqdm(amplitudes.items()):
    noise_cdfs[noise] = generate_function(f"minecraft_{noise}_sorted_vals.txt", n, noise_amplitudes)

def draw_all_noises():
    room = np.linspace(-10.0, +10.0, 1_000)
    colors = "rgbyk"
    for ind, (name, func) in enumerate(noise_cdfs.items()):
        plt.plot(room, func(room), colors[ind], label=name)
    plt.legend()
    plt.show()

def print_theoretical_ranges():
    res = {}
    for noise, noise_amplitudes in amplitudes.items():
        pre = normal_noise(None, None, None, noise_amplitudes, min_val=True)
        aft = normal_noise(None, None, None, noise_amplitudes, max_val=True)
        res[noise] = pre, aft
    print(json.dumps(res, indent=4))

def print_ranges():
    print(json.dumps({name: (func.start, func.end) for name, func in noise_cdfs.items()}, indent=4))

def print_edge_values():
    print(json.dumps({
        noise: edge_value(2.0, amplitude) for noise, amplitude in amplitudes.items()
    }, indent=4))

def print_value_factors():
    print(json.dumps({
        noise: value_factor(amplitude) for noise, amplitude in amplitudes.items()
    }, indent=4))

def test_linear_interpolation():
    x_vals = [ 0, 1, 2, 3, 4]
    y_vals = [-1, 1, 1, 2, -3]
    room = np.linspace(-1, 5, 1000)
    func = LinearInterpolation(x_vals, y_vals, -0.5, +0.5)
    plt.plot(room, func(room))
    plt.show()

def compare_with_minecraft_noise(noise_name):
    with open(f"minecraft_{noise_name}_vals.txt") as f:
        minecraft_vals = list(sorted(float(elem) for elem in f.read().strip().split("\n")))
    with open(f"{noise_name}_vals.txt") as f:
        my_vals = [float(elem) for elem in f.read().strip().split("\n")]
    minecraft_room = np.linspace(0, 1, len(minecraft_vals))
    plt.plot(minecraft_vals, minecraft_room, "g")
    my_room = np.linspace(0, 1, len(my_vals))
    plt.plot(my_vals, my_room, "r")
    plt.show()

def translate_minecraft_noise():
    for noise_name in noise_cdfs:
        with open(f"minecraft_{noise_name}_vals.txt") as f:
            vals = list(sorted([float(elem) for elem in f.read().strip().split()]))
        with open(f"minecraft_{noise_name}_sorted_vals.txt", mode="w") as f:
            f.write("\n".join(str(elem) for elem in vals))


if __name__ == "__main__":
    translate_minecraft_noise()