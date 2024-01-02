import numpy as np


def gen_mask(k_list, n, im_size):
    while True:
        Ms = []
        for k in k_list:
            N = im_size // k
            rdn = np.random.permutation(N**2)
            additive = N**2 % n
            if additive > 0:
                rdn = np.concatenate((rdn, np.asarray([-1] * (n - additive))))
            n_index = rdn.reshape(n, -1)
            for index in n_index:
                tmp = [0 if i in index else 1 for i in range(N**2)]
                tmp = np.asarray(tmp).reshape(N, N)
                tmp = tmp.repeat(k, 0).repeat(k, 1)
                Ms.append(tmp)
        yield Ms

'''    
def _reconstruct(self, mb_img: Tensor, cutout_size: int) -> Tensor:

    _, _, h, w = mb_img.shape
    num_disjoint_masks = self.cfg.params.num_disjoint_masks
    disjoint_masks = self._create_disjoint_masks((h, w), cutout_size, num_disjoint_masks)

    mb_reconst = 0
    for mask in disjoint_masks:
        mb_cutout = mb_img * mask
        mb_inpaint = self.model(mb_cutout)
        mb_reconst += mb_inpaint * (1 - mask)

    return mb_reconst

def _create_disjoint_masks(
    self,
    img_size: Tuple[int, int],
    cutout_size: int = 8,               # cutout_size：论文中的网格大小k
    num_disjoint_masks: int = 3,        # num_disjoint_masks：mask的个数，即Si的个数k
) -> List[Tensor]:

    img_h, img_w = img_size
    grid_h = math.ceil(img_h / cutout_size)
    grid_w = math.ceil(img_w / cutout_size)
    num_grids = grid_h * grid_w         # num_grids：总的网格数N
    disjoint_masks = []                 # array_split：返回一堆拆分的数组，（数组，份数）
    for grid_ids in np.array_split(np.random.permutation(num_grids), num_disjoint_masks):
        flatten_mask = np.ones(num_grids)   # np.ones返回全是1的数组
        flatten_mask[grid_ids] = 0
        mask = flatten_mask.reshape((grid_h, grid_w))
        mask = mask.repeat(cutout_size, axis=0).repeat(cutout_size, axis=1)
        mask = torch.tensor(mask, requires_grad=False, dtype=torch.float)
        mask = mask.to(self.cfg.params.device)
        disjoint_masks.append(mask)

    return disjoint_masks
'''