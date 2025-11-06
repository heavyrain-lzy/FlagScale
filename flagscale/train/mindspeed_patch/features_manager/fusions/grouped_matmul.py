from mindspeed_patch.features_manager.feature import MindSpeedFeature


class GroupedMatmulFeature(MindSpeedFeature):
    def __init__(self):
        super().__init__('grouped-matmul', optimization_level=0)

    def register_patches(self, patch_manager, args):
        from mindspeed_patch.core.fusions.grouped_matmul import (
            Ops,
            assert_grouped_gemm_is_available,
            get_device_capability,
            grouped_gemm_is_available,
        )

        patch_manager.register_patch('megatron.core.transformer.moe.grouped_gemm_util.ops', Ops)
        patch_manager.register_patch(
            'megatron.core.transformer.moe.grouped_gemm_util.grouped_gemm_is_available',
            grouped_gemm_is_available,
        )
        patch_manager.register_patch(
            'megatron.core.transformer.moe.grouped_gemm_util.assert_grouped_gemm_is_available',
            assert_grouped_gemm_is_available,
        )
        patch_manager.register_patch('torch.cuda.get_device_capability', get_device_capability)
