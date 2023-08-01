import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from transformers import BatchEncoding

import matplotlib.pyplot as plt
from matplotlib.patches import Circle, RegularPolygon
from matplotlib.path import Path
from matplotlib.projections.polar import PolarAxes
from matplotlib.projections import register_projection
from matplotlib.spines import Spine
from matplotlib.transforms import Affine2D


def get_device():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    return device


def score_text(model, tokenizer, story):

    device = get_device()

    tokenized = tokenizer(story, is_split_into_words=True)
    
    input_ids = tokenized["input_ids"][1:]
    batch_size = int(np.ceil(len(input_ids)/model.config.max_length))
    to_pad = model.config.max_length * batch_size - len(input_ids)
    
    # batched input ids
    input_ids = input_ids + [tokenizer.pad_token_id for _ in range(to_pad-batch_size)]
    input_ids = torch.LongTensor(input_ids).reshape(batch_size, model.config.max_length-1)

    # batched attention mask
    attention_mask = tokenized["attention_mask"][1:]
    attention_mask = attention_mask + [0 for _ in range(to_pad-batch_size)]
    attention_mask = torch.LongTensor(attention_mask).reshape(batch_size, model.config.max_length-1)

    batch = BatchEncoding({
        "input_ids": torch.stack(
                [torch.concat([torch.LongTensor([tokenizer.eos_token_id]), t]).to(device) for t in input_ids]
            ),
        "labels": torch.stack(
                [torch.concat([torch.LongTensor([tokenizer.eos_token_id]), t]).to(device) for t in input_ids]
            ),
        "attention_mask": torch.stack(
                [torch.concat([torch.LongTensor([1]), t]).to(device) for t in input_ids]
            )
        
    })
    
    assert batch["input_ids"].shape ==  (batch_size, model.config.max_length), batch["input_ids"].shape

    # inference
    with torch.no_grad():
        output = model(**batch)

    # calculate word-level surprisal
    words, word_surprisal = [], []

    curr_word_ix = 0
    curr_word_surp = []
    curr_toks = ""

    for logits, input_ids in zip(output.logits, batch["input_ids"]):
        output_ids = input_ids[1:]
        tokens = [tok for tok in tokenizer.convert_ids_to_tokens(output_ids) if tok != tokenizer.pad_token]
        indices = torch.arange(0, output_ids.shape[0]).to(device)
        surprisal = -1*torch.log2(F.softmax(logits, dim=-1)).squeeze(0)[indices, output_ids]

        for i in range(0, len(tokens)):
            # necessary for diacritics in Dundee
            cleaned_tok = tokens[i].replace("Ġ", "", 1).encode("latin-1").decode("utf-8")
            # for word-level surprisal
            curr_word_surp.append(surprisal[i].item())
            curr_toks += cleaned_tok

            # summing subword token surprisal ("rolling")
            story[curr_word_ix] = story[curr_word_ix].replace(cleaned_tok, "", 1)
            if story[curr_word_ix] == "":
                words.append(curr_toks)
                word_surprisal.append(np.round(sum(curr_word_surp),4))
                curr_word_surp = []
                curr_toks = ""
                curr_word_ix += 1

    assert len(words) == len(story), f"len(story)={len(story)} != len(words)={len(words)}"
    
    return pd.DataFrame({
        "word": words,
        "surprisal": word_surprisal
    })


def print_latex_table(latex_table_content):

    task0 = list(latex_table_content.keys())[0]

    configs_short = [config for config in latex_table_content[task0]]

    print("\\begin{tabular}{l|c " + " c "*24 + "}")
    print("\hline")

    header = "\\textbf{Task} & " +  " & ".join([f"\\textbf{{{config}}}" for config in configs_short]) + " \\\\"
    
    print(header)
    print("\hline\hline")

    for task, row in latex_table_content.items():
        if task != "average":
            task_str = f"{task.replace('_', '-')} & " + " & ".join([latex_table_field for latex_table_field in row.values()]) + " \\\\"
            print(task_str)

    print("\hline")
    
    avg_str = "\\textbf{Average} & " + " & ".join([latex_table_field for latex_table_field in latex_table_content["average"].values()]) + " \\\\"
    print(avg_str)
    print("\hline")
    print("\\end{tabular}")


# code for radar chart taken from: https://matplotlib.org/stable/gallery/specialty_plots/radar_chart.html

def radar_factory(num_vars, frame='circle'):
    """
    Create a radar chart with `num_vars` axes.

    This function creates a RadarAxes projection and registers it.

    Parameters
    ----------
    num_vars : int
        Number of variables for radar chart.
    frame : {'circle', 'polygon'}
        Shape of frame surrounding axes.

    """
    # calculate evenly-spaced axis angles
    theta = np.linspace(0, 2*np.pi, num_vars, endpoint=False)

    class RadarTransform(PolarAxes.PolarTransform):

        def transform_path_non_affine(self, path):
            # Paths with non-unit interpolation steps correspond to gridlines,
            # in which case we force interpolation (to defeat PolarTransform's
            # autoconversion to circular arcs).
            if path._interpolation_steps > 1:
                path = path.interpolated(num_vars)
            return Path(self.transform(path.vertices), path.codes)

    class RadarAxes(PolarAxes):

        name = 'radar'
        PolarTransform = RadarTransform

        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            # rotate plot such that the first axis is at the top
            self.set_theta_zero_location('N')

        def fill(self, *args, closed=True, **kwargs):
            """Override fill so that line is closed by default"""
            return super().fill(closed=closed, *args, **kwargs)

        def plot(self, *args, **kwargs):
            """Override plot so that line is closed by default"""
            lines = super().plot(*args, **kwargs)
            for line in lines:
                self._close_line(line)

        def _close_line(self, line):
            x, y = line.get_data()
            # FIXME: markers at x[0], y[0] get doubled-up
            if x[0] != x[-1]:
                x = np.append(x, x[0])
                y = np.append(y, y[0])
                line.set_data(x, y)

        def set_varlabels(self, labels):
            self.set_thetagrids(np.degrees(theta), labels)

        def _gen_axes_patch(self):
            # The Axes patch must be centered at (0.5, 0.5) and of radius 0.5
            # in axes coordinates.
            if frame == 'circle':
                return Circle((0.5, 0.5), 0.5)
            elif frame == 'polygon':
                return RegularPolygon((0.5, 0.5), num_vars,
                                      radius=.5, edgecolor="k")
            else:
                raise ValueError("Unknown value for 'frame': %s" % frame)

        def _gen_axes_spines(self):
            if frame == 'circle':
                return super()._gen_axes_spines()
            elif frame == 'polygon':
                # spine_type must be 'left'/'right'/'top'/'bottom'/'circle'.
                spine = Spine(axes=self,
                              spine_type='circle',
                              path=Path.unit_regular_polygon(num_vars))
                # unit_regular_polygon gives a polygon of radius 1 centered at
                # (0, 0) but we want a polygon of radius 0.5 centered at (0.5,
                # 0.5) in axes coordinates.
                spine.set_transform(Affine2D().scale(.5).translate(.5, .5)
                                    + self.transAxes)
                return {'polar': spine}
            else:
                raise ValueError("Unknown value for 'frame': %s" % frame)

    register_projection(RadarAxes)
    return theta
