def csls(k, img_embs, txt_embs):
    """
    implementation following this paper:
    https://arxiv.org/pdf/1710.04087.pdf
    """
    sims_matrix = img_embs @ txt_embs.t()

    topk_sim_0, _ = sims_matrix.topk(k=k, dim=0)
    topk_sim_0 = topk_sim_0.mean(dim=0, keepdim=True)

    topk_sim_1, _ = sims_matrix.topk(k=k, dim=1)
    topk_sim_1 = topk_sim_1.mean(dim=1, keepdim=True)

    sims_matrix = 2 * sims_matrix - topk_sim_0 - topk_sim_1

    return sims_matrix
