import SimpleITK as sitk
import sys
import os

def demons_registration(
    fixed_image, moving_image, fixed_points=None, moving_points=None
):
    registration_method = sitk.ImageRegistrationMethod()

    # Create initial identity transformation.
    transform_to_displacment_field_filter = sitk.TransformToDisplacementFieldFilter()
    transform_to_displacment_field_filter.SetReferenceImage(fixed_image)
    # The image returned from the initial_transform_filter is transferred to the transform and cleared out.
    initial_transform = sitk.DisplacementFieldTransform(
        transform_to_displacment_field_filter.Execute(sitk.Transform())
    )

    # Regularization (update field - viscous, total field - elastic).
    initial_transform.SetSmoothingGaussianOnUpdate(
        varianceForUpdateField=0.0, varianceForTotalField=2.0
    )

    registration_method.SetInitialTransform(initial_transform)

    # registration_method.SetMetricAsDemons(
    #     10
    # )  # intensities are equal if the difference is less than 10HU
    registration_method.SetMetricAsCorrelation()

    # Multi-resolution framework.
    registration_method.SetShrinkFactorsPerLevel(shrinkFactors=[4, 2, 1])
    registration_method.SetSmoothingSigmasPerLevel(smoothingSigmas=[8, 4, 0])

    registration_method.SetInterpolator(sitk.sitkLinear)
    # If you have time, run this code as is, otherwise switch to the gradient descent optimizer
    # registration_method.SetOptimizerAsConjugateGradientLineSearch(learningRate=1.0, numberOfIterations=20, convergenceMinimumValue=1e-6, convergenceWindowSize=10)

    registration_method.SetOptimizerAsGradientDescent(
        learningRate=1e-2,
        numberOfIterations=500,
        convergenceMinimumValue=1e-6,
        convergenceWindowSize=10,
    )
    registration_method.SetOptimizerScalesFromPhysicalShift()
    registration_method.AddCommand(sitk.sitkIterationEvent, lambda: command_iteration(registration_method))

    # If corresponding points in the fixed and moving image are given then we display the similarity metric
    # and the TRE during the registration.
    if fixed_points and moving_points:
        registration_method.AddCommand(
            sitk.sitkStartEvent, rc.metric_and_reference_start_plot
        )
        registration_method.AddCommand(
            sitk.sitkEndEvent, rc.metric_and_reference_end_plot
        )
        registration_method.AddCommand(
            sitk.sitkIterationEvent,
            lambda: rc.metric_and_reference_plot_values(
                registration_method, fixed_points, moving_points
            ),
        )

    outTx = registration_method.Execute(fixed_image, moving_image)
    return registration_method, outTx

def command_iteration(method):
    print(
        f"{method.GetOptimizerIteration():3} "
        + f"= {method.GetMetricValue():10.5f} "
        + f": {method.GetOptimizerPosition()[:3]}"
    )

## see the result
import matplotlib.pyplot as plt
from monai.visualize import matshow3d
import numpy as np
from typing import List
def save_imgs(imgs: List[np.ndarray], path, cmap: str = "gray"):
    fig = plt.figure(figsize=(10, 10))
    le = len(imgs)
    axs = fig.subplots(2, le // 2)
    plt.subplots_adjust(wspace=0.01, hspace=0.01, left=0.01, right=0.99, top=0.99, bottom=0.01)
    for img, ax in zip(imgs, axs.flat):
        matshow3d(img, fig = ax, cmap=cmap, show=False)
    for ax in axs.flat[le:]:
        ax.axis('off')
    fig.savefig(path)

def normal_registration(fixed, moving):
    R = sitk.ImageRegistrationMethod()
    # R.SetMetricAsMeanSquares()
    R.SetMetricAsCorrelation()

    # R.SetOptimizerAsRegularStepGradientDescent(4.0, 0.01, 200)
    # R.SetInitialTransform(sitk.TranslationTransform(fixed.GetDimension()))
    R.SetOptimizerAsRegularStepGradientDescent(
        learningRate=2.0,
        minStep=1e-4,
        numberOfIterations=500,
        gradientMagnitudeTolerance=1e-8,
    )
    R.SetOptimizerScalesFromIndexShift()
    tx = sitk.CenteredTransformInitializer(
        fixed, moving, sitk.AffineTransform(3)
    )
    R.SetInitialTransform(tx)

    R.SetInterpolator(sitk.sitkLinear)
    R.AddCommand(sitk.sitkIterationEvent, lambda: command_iteration(R))

    outTx = R.Execute(fixed, moving)
    return R, outTx

def get_correlation(i1: np.ndarray, i2: np.ndarray):
    """negative normalized cross correlation image metric."""
    i1_hat = (i1 - i1.mean()) / i1.std()
    i2_hat = (i2 - i2.mean()) / i2.std()
    cor = np.sum(i1_hat * i2_hat) / (i1.size - 1)
    return -cor

import SimpleITK as sitk
import torch
def wrap_normal(fixed: np.ndarray, moving: np.ndarray):
    fixed_img = sitk.GetImageFromArray(fixed)
    moving_img = sitk.GetImageFromArray(moving)
    # set image to sitkFloat32
    fixed_img = sitk.Cast(fixed_img, sitk.sitkFloat32)
    moving_img = sitk.Cast(moving_img, sitk.sitkFloat32)
    R, outTx = normal_registration(fixed_img, moving_img)

    gx, gy, gz = np.mgrid[0:fixed_img.GetSize()[0], 0:fixed_img.GetSize()[1], 0:fixed_img.GetSize()[2]]
    points = np.stack([gx, gy, gz], axis=-1, dtype=np.float64)

    def get_aff_(tx):
        center = np.array(tx.GetCenter())
        trans = np.array(tx.GetTranslation())
        A = np.array(tx.GetMatrix()).reshape(3, 3)
        offset = -A@center + trans + center
        aff_ = lambda x: A@(x-center) + trans + center
        return aff_
    def get_flow_(aff_):
        reg_points = np.apply_along_axis(aff_, 3, points)
        flow = reg_points - points
        flow = flow[...,[2,1,0]].transpose(2,1,0,3).copy()
        return flow
    aff_ = get_aff_(outTx)
    inv_aff_ = get_aff_(outTx.GetInverse())
    t = []

    flow = get_flow_(aff_)
    # from monai.networks.blocks import Warp
    # out = Warp(padding_mode="border", mode="bilinear")(torch.from_numpy(moving).unsqueeze(0).unsqueeze(0).float(), torch.from_numpy(flow).float().unsqueeze(0).permute(0, 4, 1, 2, 3))
    # import sys
    # sys.path.insert('..')
    # print("before registration", get_correlation(fixed, moving))
    # print("after registration", get_correlation(fixed, out.squeeze(0).squeeze(0).numpy()))

    return flow

def main(args):
    # if len(args) < 3:
    #     print(
    #         "Usage:",
    #         "ImageRegistrationMethod1",
    #         "<fixedImageFilter> <movingImageFile>",
    #         "<outputTransformFile>",
    #     )
    #     sys.exit(1)

    fixed = sitk.ReadImage("/home/duhao/registration/recursive_cascaded_networks/eval/intflow/example_data/orig_AD001_L.img", sitk.sitkFloat32)

    moving = sitk.ReadImage("/home/duhao/registration/recursive_cascaded_networks/eval/intflow/example_data/orig_NL001_L.img", sitk.sitkFloat32)
    fixed = sitk.GetImageFromArray(sitk.GetArrayFromImage(fixed))
    moving = sitk.GetImageFromArray(sitk.GetArrayFromImage(moving))

    wrap_normal(sitk.GetArrayFromImage(fixed), sitk.GetArrayFromImage(moving))

    R, outTx = normal_registration(fixed, moving)
    # R, outTx = demons_registration(fixed, moving)
    # breakpoint()

    print("-------")
    print(outTx)
    print(f"Optimizer stop condition: {R.GetOptimizerStopConditionDescription()}")
    print(f" Iteration: {R.GetOptimizerIteration()}")
    print(f" Metric value: {R.GetMetricValue()}")

    # sitk.WriteTransform(outTx, args[3])

    resampler = sitk.ResampleImageFilter()
    resampler.SetReferenceImage(fixed)
    resampler.SetInterpolator(sitk.sitkLinear)
    resampler.SetDefaultPixelValue(0)
    resampler.SetTransform(outTx)

    out = resampler.Execute(moving)
    print("before registration", get_correlation(sitk.GetArrayFromImage(fixed), sitk.GetArrayFromImage(moving)))
    print("after registration", get_correlation(sitk.GetArrayFromImage(fixed), sitk.GetArrayFromImage(out)))

    save_imgs([sitk.GetArrayFromImage(fixed), sitk.GetArrayFromImage(out), sitk.GetArrayFromImage(moving)], "test.png")

    simg1 = sitk.Cast(sitk.RescaleIntensity(fixed), sitk.sitkUInt8)
    simg2 = sitk.Cast(sitk.RescaleIntensity(out), sitk.sitkUInt8)
    simg_mov = sitk.Cast(sitk.RescaleIntensity(moving), sitk.sitkUInt8)
    cimg = sitk.Compose(simg1, simg2, simg1 // 2.0 + simg2 // 2.0)
    # return_images = {"fixed": fixed,
    #                  "moving": moving,
    #                  "composition": cimg}
    # return return_images

if __name__ == "__main__":
    main(sys.argv[1:])