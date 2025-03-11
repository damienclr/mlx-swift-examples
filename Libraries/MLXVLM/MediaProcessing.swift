// Copyright © 2024 Apple Inc.

import AVFoundation
import CoreImage.CIFilterBuiltins
import MLX
import MLXLMCommon

private let context = CIContext()

/// Collection of methods for processing media (images, video, etc.).
///
/// A typical image preparation pipeline might look like this:
///
/// ```swift
/// var image: CIImage
/// image = MediaProcessing.inSRGBToneCurveSpace(image)
///
/// // Apply user instructions
/// image = MediaProcessing.apply(image, processing: processing)
///
/// image = MediaProcessing.resampleBicubic(image, to: config.size.cgSize)
/// image = MediaProcessing.normalize(
///     image, mean: config.imageMeanTuple, std: config.imageStdTuple)
///
/// return MediaProcessing.asMLXArray(image)
/// ```
///
/// This is the responsibility of the `UserInputProcessor`.
public enum MediaProcessing {

    /// VLM media processing is normally done withut regard to the colorspace.  Many,
    /// though not all, images are stored in sRGB and this wiill be the implicit colorspace
    /// used.  This converts to a colorspace with an sRGB tone curve, though not necessarily
    /// sRGB primaries, etc.
    ///
    /// See ``inLinearToneCurveSpace(_:)``
    static public func inSRGBToneCurveSpace(_ image: CIImage) -> CIImage {
        let filter = CIFilter.linearToSRGBToneCurve()
        filter.inputImage = image
        return filter.outputImage!
    }

    /// Inverse of ``inSRGBToneCurveSpace(_:)`` (for completeness).
    static public func inLinearToneCurveSpace(_ image: CIImage) -> CIImage {
        let filter = CIFilter.sRGBToneCurveToLinear()
        filter.inputImage = image
        return filter.outputImage!
    }

    /// Compute the best fit size of one size in another (respecting aspect ratio).
    static public func bestFit(_ size: CGSize, in other: CGSize) -> CGSize {
        let scale = bestFitScale(size, in: other)
        return CGSize(width: round(size.width * scale), height: round(size.height * scale))
    }

    /// Compute the best fit scale of one size in another (respecting aspect ratio).
    static public func bestFitScale(_ size: CGSize, in other: CGSize) -> CGFloat {
        min(other.width / size.width, other.height / size.height)
    }

    enum MediaProcessingError: LocalizedError {
        case transformFailed

        var errorDescription: String? {
          switch self {
            case .transformFailed: "Failed to transform image"
          }
        }
    }

    /// Resample the image using bicubic interpolation.
    /// - Parameters:
    ///   - image: The image to resample
    ///   - size: The target size
    /// - Returns: The resampled image
    public static func resampleBicubic(_ image: CIImage, to size: CGSize) throws -> CIImage {
        // Create a bicubic scale filter
        let filter = CIFilter.bicubicScaleTransform()
        filter.inputImage = image
        filter.scale = Float(size.width / image.extent.width)
        filter.aspectRatio = 1.0
        guard let scaledImage = filter.outputImage else {
            throw MediaProcessingError.transformFailed
        }
        // Calculate the crop rect to get exactly the requested size
        // Scale height separately to match the target height
        let heightScale = size.height / scaledImage.extent.height
        let finalImage = scaledImage.transformed(by: CGAffineTransform(scaleX: 1.0, y: heightScale))
        // Create a rect with the exact dimensions we want
        let exactRect = CGRect(
            x: 0,
            y: 0,
            width: size.width,
            height: size.height
        )
        // Crop to ensure exact dimensions
        return finalImage.cropped(to: exactRect)
    }

    /// Normalize the image using the given mean and standard deviation parameters.
    static public func normalize(
        _ image: CIImage, mean: (CGFloat, CGFloat, CGFloat), std: (CGFloat, CGFloat, CGFloat)
    ) -> CIImage {
        let filter = CIFilter.colorMatrix()
        filter.inputImage = image

        // This should match
        // https://pytorch.org/vision/main/generated/torchvision.transforms.Normalize.html
        //
        // output[channel] = (input[channel] - mean[channel]) / std[channel]
        //
        // The CI filter computes input * factor + bias so we want to do:
        // input * 1 / std - mean / std

        filter.rVector = .init(x: 1 / std.0, y: 0, z: 0, w: 0)
        filter.gVector = .init(x: 0, y: 1 / std.1, z: 0, w: 0)
        filter.bVector = .init(x: 0, y: 0, z: 1 / std.2, w: 0)

        filter.aVector = .init(x: 0, y: 0, z: 0, w: 1)
        filter.biasVector = .init(x: -mean.0 / std.0, y: -mean.1 / std.1, z: -mean.2 / std.2, w: 0)

        return filter.outputImage!
    }

    /// Convert the CIImage into a planar 3 channel MLXArray `[1, C, H, W]`
    /// - Parameters:
    ///   - image: The image to convert
    ///   - colorSpace: Optional color space for rendering
    /// - Returns: The MLXArray representation of the image
    static public func asMLXArray(_ image: CIImage, colorSpace: CGColorSpace? = nil) -> MLXArray {
        let size = image.extent.size
        let w = Int(size.width.rounded())
        let h = Int(size.height.rounded())

        // probably not strictly necessary, but this is what happens in
        // e.g. image_processing_siglip in transformers (float32)
        let format = CIFormat.RGBAf
        let componentsPerPixel = 4
        let bytesPerPixel = componentsPerPixel * 4
        let bytesPerRow = w * bytesPerPixel

        var data = Data(count: w * h * bytesPerPixel)
        data.withUnsafeMutableBytes { ptr in
            context.render(
                image, toBitmap: ptr.baseAddress!, rowBytes: bytesPerRow, bounds: image.extent,
                format: format, colorSpace: colorSpace)
            context.clearCaches()
        }

        var array = MLXArray(data, [h, w, 4], type: Float32.self)

        // Drop 4th channel
        array = array[0..., 0..., ..<3]

        // Convert to 1, C, H, W
        array = array.reshaped(1, h, w, 3).transposed(0, 3, 1, 2)

        return array
    }

    /// Apply `UserInput.Processing`, if needed, to the image.
    static func apply(_ image: CIImage, processing: UserInput.Processing?) -> CIImage {
        var image = image

        if let resize = processing?.resize {
            let scale = bestFitScale(image.extent.size, in: resize)
            image = image.transformed(by: CGAffineTransform(scaleX: scale, y: scale))
        }

        return image
    }

    static func asCIImageSequence(_ asset: AVAsset, samplesPerSecond: Int) async throws -> [CIImage]
    {
        // Use AVAssetImageGenerator to extract frames
        let generator = AVAssetImageGenerator(asset: asset)
        generator.appliesPreferredTrackTransform = true
        generator.requestedTimeToleranceBefore = .zero
        generator.requestedTimeToleranceAfter = .zero

        // Calculate the time values we want to sample
        guard let duration = try? await asset.load(.duration) else {
            throw NSError(
                domain: "MediaProcessing", code: -1,
                userInfo: [NSLocalizedDescriptionKey: "Failed to load the asset's duration"])
        }

        let durationInSeconds = duration.seconds
        let samplesPerSecond = Double(samplesPerSecond)
        let secondsPerSample = 1.0 / samplesPerSecond
        let totalFramesToSample = durationInSeconds * samplesPerSecond
        let durationTimeValue = duration.value
        let sampledTimeValues = MLXArray.linspace(
            0, durationTimeValue, count: Int(totalFramesToSample)
        ).asArray(Int64.self)

        // Construct a CMTime using the sampled CMTimeValue's and the asset's timescale
        let timescale = duration.timescale
        let sampledTimes = sampledTimeValues.map { CMTime(value: $0, timescale: timescale) }

        // Collect the frames
        var ciImages: [CIImage] = []
        for await result in await generator.images(for: sampledTimes) {
            switch result {
            case .success(requestedTime: let requested, let image, actualTime: let actual):
                let ciImage = CIImage(
                    cgImage: image, options: [.colorSpace: CGColorSpace(name: CGColorSpace.sRGB)!])
                ciImages.append(ciImage)
            case .failure(requestedTime: let requested, let error):
                break
            }
        }

        return ciImages
    }
}
