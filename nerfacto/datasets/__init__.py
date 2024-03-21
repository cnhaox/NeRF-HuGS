import datasets.blender as blender
import datasets.llff as llff
import datasets.phototourism as phototourism
import datasets.kubric as kubric
import datasets.distractor as distractor

dataset_dict = {
    'blender': blender.BlenderDataset,
    'llff': llff.LLFFDataset,
    'phototourism': phototourism.PhototourismDataset,
    'kubric': kubric.KubricDataset,
    'distractor': distractor.DistractorDataset,
}