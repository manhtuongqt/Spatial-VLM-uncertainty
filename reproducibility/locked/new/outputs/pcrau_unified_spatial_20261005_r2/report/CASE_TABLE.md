# Case thật trên dev

Masks/nhãn chỉ dùng hậu kiểm. Geometry compatibility không xác nhận identity.

| Case dev | Target → quan hệ → anchor | MAP target / anchor | Margin px / compatibility | Risk / action | Hậu kiểm |
|---|---|---|---|---|---|
| `000001__clean` | apple → right_of → purple cube | [350, 150] / [290, 250] | 48 / 0.999901 | 0.202758 / EXECUTE | FOUND; anchor=True; pixels=3647 |
| `000001__relation_counterfactual` | purple cube → left_of → apple | [270, 130] / [350, 150] | 68 / 0.998641 | 0.144897 / EXECUTE | FOUND; anchor=True; pixels=4941 |
| `000001__depth_corruption` | apple → right_of → purple cube | [350, 150] / [290, 250] | 48 / 0.999884 | 0.970546 / REOBSERVE | INSUFFICIENT_EVIDENCE; anchor=True; pixels=3647 |
| `000383__relation_counterfactual` | yellow cube → left_of → apple | [270, 310] / [290, 170] | 8 / 0.958688 | 0.152992 / EXECUTE | FOUND; anchor=False; pixels=4453 |
| `000042__clean` | blue cube → left_of → green cube | [350, 210] / [350, 230] | -12 / 0.300290 | 0.280246 / EXECUTE | ABSENT; anchor=False; pixels=2056 |
| `000392__clean` | orange cube → left_of → orange | [130, 170] / [130, 170] | -12 / 0.388176 | 0.937834 / ABSTAIN | ABSENT; anchor=None; pixels=0 |
| `000001__semantic_counterfactual` | pink cube → direct → — | [270, 130] / None | None / None | 0.026674 / EXECUTE | FOUND |
| `000017__clean` | unparsed → unsupported → — | [330, 210] / None | None / None | 0.774808 / REOBSERVE | FOUND |
