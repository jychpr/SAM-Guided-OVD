import os

def patch_model_signatures():
    # 1. Patch ov_dquo.py
    file1 = "models/ov_dquo/ov_dquo.py"
    with open(file1, "r") as f:
        content = f.read()

    if "sam_proposals=None" not in content:
        content = content.replace(
            "targets=None,\n    ):", 
            "targets=None,\n        sam_proposals=None,\n    ):"
        )
        content = content.replace(
            "backbone=self.backbone,\n        )", 
            "backbone=self.backbone,\n            sam_proposals=sam_proposals,\n        )"
        )
        with open(file1, "w") as f:
            f.write(content)
        print(f"[*] Successfully patched {file1}")
    else:
        print(f"[-] {file1} already patched.")

    # 2. Patch ov_deformable_transformer.py
    file2 = "models/transformer/ov_deformable_transformer.py"
    with open(file2, "r") as f:
        content = f.read()

    if "sam_proposals=None" not in content:
        content = content.replace(
            "targets=None,\n        backbone=None\n    ):",
            "targets=None,\n        backbone=None,\n        sam_proposals=None\n    ):"
        )
        
        original_logic = """            topk = self.num_queries
            topk_proposals = torch.topk(enc_outputs_class_unselected.max(-1)[0], topk, dim=1)[1]
            # gather boxes
            refpoint_embed_undetach = torch.gather(enc_outputs_coord_unselected,1,topk_proposals.unsqueeze(-1).repeat(1, 1, 4),)  # unsigmoid
            refpoint_embed_ = refpoint_embed_undetach.detach()
            init_box_proposal = torch.gather(output_proposals, 1, topk_proposals.unsqueeze(-1).repeat(1, 1, 4)).sigmoid()  
            # gather tgt
            tgt_undetach = torch.gather(output_memory,1,topk_proposals.unsqueeze(-1).repeat(1, 1, self.d_model),)
            if self.embed_init_tgt:
                tgt_ = (self.tgt_embed.weight[:, None, :].repeat(1, bs, 1).transpose(0, 1)) 
            else:
                tgt_ = tgt_undetach.detach()"""

        thesis_logic = """            topk = self.num_queries
            if sam_proposals is not None:
                # --- THESIS INJECTION: FastSAM PRIORS ---
                init_box_proposal = sam_proposals
                from util.misc import inverse_sigmoid
                refpoint_embed_ = inverse_sigmoid(sam_proposals)
                # Initialize object queries with zeros; cross-attention will populate them based on SAM geometry
                tgt_ = torch.zeros(bs, sam_proposals.size(1), self.d_model, device=output_memory.device)
                # ----------------------------------------
            else:
                topk_proposals = torch.topk(enc_outputs_class_unselected.max(-1)[0], topk, dim=1)[1]
                # gather boxes
                refpoint_embed_undetach = torch.gather(enc_outputs_coord_unselected,1,topk_proposals.unsqueeze(-1).repeat(1, 1, 4),)  # unsigmoid
                refpoint_embed_ = refpoint_embed_undetach.detach()
                init_box_proposal = torch.gather(output_proposals, 1, topk_proposals.unsqueeze(-1).repeat(1, 1, 4)).sigmoid()  
                # gather tgt
                tgt_undetach = torch.gather(output_memory,1,topk_proposals.unsqueeze(-1).repeat(1, 1, self.d_model),)
                if self.embed_init_tgt:
                    tgt_ = (self.tgt_embed.weight[:, None, :].repeat(1, bs, 1).transpose(0, 1)) 
                else:
                    tgt_ = tgt_undetach.detach()"""

        content = content.replace(original_logic, thesis_logic)
        with open(file2, "w") as f:
            f.write(content)
        print(f"[*] Successfully patched {file2}")
    else:
        print(f"[-] {file2} already patched.")

if __name__ == "__main__":
    patch_model_signatures()