import CoreImage
import Foundation
import Hub
import MLX
import MLXFast
import MLXLMCommon
import MLXNN
import Tokenizers

// MARK: - Common Utilities for Qwen 2 VL and Qwen 2.5 VL

public struct QwenVL {
    /// Rotates half the hidden dims of the input
    static func rotateHalf(_ x: MLXArray) -> MLXArray {
        let index = x.dim(-1) / 2
        let x1 = x[.ellipsis, 0 ..< index]
        let x2 = x[.ellipsis, index...]
        return concatenated([-x2, x1], axis: -1)
    }

    static func mergeInputIdsWithImageFeatures(
        inputIds: MLXArray, inputEmbeds: MLXArray, imageFeatures: MLXArray,
        imageTokenId: Int, videoTokenId: Int
    ) -> MLXArray {
        var imageIndices = [Int]()
        for (i, v) in inputIds.asArray(Int.self).enumerated() {
            if v == imageTokenId || v == videoTokenId {
                imageIndices.append(i)
            }
        }

        // Make sure shapes match before assignment
        var result = inputEmbeds
        if result.ndim == 2 {
            result = result[.newAxis, 0..., 0...]
        }

        if imageFeatures.ndim == 2 {
            let reshapedFeatures = imageFeatures[.newAxis, 0..., 0...]
            result[0..., MLXArray(imageIndices), 0...] = reshapedFeatures
        } else {
            result[0..., MLXArray(imageIndices), 0...] = imageFeatures
        }

        return result
    }

    public class VisionRotaryEmbedding {
        let dimensions: Int
        let theta: Float
        let inverseFreq: MLXArray

        init(dimensions: Int, theta: Float) {
            self.dimensions = dimensions
            self.theta = theta
            let p = MLXArray(stride(from: 0, to: dimensions, by: 2)).asType(.float32) / dimensions
            self.inverseFreq = 1.0 / pow(theta, p)
        }

        func callAsFunction(sequenceLength: Int) -> MLXArray {
            let seq = MLXArray(0 ..< sequenceLength).asType(inverseFreq.dtype)
            let freqs = outer(seq, inverseFreq)
            return freqs
        }
    }

    public class PatchEmbed: Module, UnaryLayer {
        @ModuleInfo var proj: Conv3d

        let patchSize: Int
        let temporalPatchSize: Int
        let inChannels: Int
        let outputDimensions: Int

        // For Qwen 2 VL
        convenience init(
            patchSize: Int, temporalPatchSize: Int, inChannels: Int, embedDimensions: Int
        ) {
            self.init(
                patchSize: patchSize, temporalPatchSize: temporalPatchSize,
                inChannels: inChannels, outputDimensions: embedDimensions)
        }

        // For Qwen 2.5 VL
        convenience init(patchSize: Int, temporalPatchSize: Int, inChannels: Int, hiddenSize: Int) {
            self.init(
                patchSize: patchSize, temporalPatchSize: temporalPatchSize,
                inChannels: inChannels, outputDimensions: hiddenSize)
        }

        // Common initializer
        init(patchSize: Int, temporalPatchSize: Int, inChannels: Int, outputDimensions: Int) {
            self.patchSize = patchSize
            self.temporalPatchSize = temporalPatchSize
            self.inChannels = inChannels
            self.outputDimensions = outputDimensions

            let kernelSize = IntOrTriple([temporalPatchSize, patchSize, patchSize])
            self._proj.wrappedValue = Conv3d(
                inputChannels: inChannels,
                outputChannels: outputDimensions,
                kernelSize: kernelSize,
                stride: kernelSize,
                bias: false
            )
        }

        public func callAsFunction(_ hiddenStates: MLXArray) -> MLXArray {
            var hiddenStates = hiddenStates.reshaped(
                -1, inChannels, temporalPatchSize, patchSize, patchSize
            ).movedAxis(source: 1, destination: 4)

            hiddenStates = proj(hiddenStates)
            hiddenStates = hiddenStates.reshaped(-1, outputDimensions)
            return hiddenStates
        }
    }

    // image_processing_qwen2_vl.smart_resize
    static func targetSize(height: Int, width: Int, factor: Int, minPixels: Int, maxPixels: Int)
        throws
        -> (Int, Int)
    {
        if height < factor {
            throw VLMError.imageProcessingFailure(
                "Height: \(height) must be larger than factor: \(factor)")
        }
        if width < factor {
            throw VLMError.imageProcessingFailure(
                "Width: \(width) must be larger than factor: \(factor)")
        }
        if max(height, width) / min(height, width) > 200 {
            throw VLMError.imageProcessingFailure(
                "Absolute aspect ratio must be smaller than 200: \(width) × \(height)")
        }

        // Maximum allowed dimension for any single side to prevent buffer overflows
        // This is important for portrait/landscape images with extreme aspect ratios
        let maxDimension = 224

        var hBar = max(factor, Int(round(Float(height) / Float(factor))) * factor)
        var wBar = max(factor, Int(round(Float(width) / Float(factor))) * factor)

        // Start by scaling based on total pixel count
        if hBar * wBar > maxPixels {
            let beta = sqrt(Float(height * width) / Float(maxPixels))
            hBar = Int(floor(Float(height) / beta / Float(factor))) * factor
            wBar = Int(floor(Float(width) / beta / Float(factor))) * factor
        } else if hBar * wBar < minPixels {
            let beta = sqrt(Float(minPixels) / Float(height * width))
            hBar = Int(ceil(Float(height) * beta / Float(factor))) * factor
            wBar = Int(ceil(Float(width) * beta / Float(factor))) * factor
        }

        // Additionally check if either dimension exceeds the maximum allowed
        if hBar > maxDimension {
            // Calculate how much we need to scale down height
            let scale = Float(maxDimension) / Float(hBar)
            // Apply that scale to both dimensions to maintain aspect ratio
            hBar = Int(round(Float(hBar) * scale / Float(factor))) * factor
            wBar = Int(round(Float(wBar) * scale / Float(factor))) * factor
        }

        if wBar > maxDimension {
            // Calculate how much we need to scale down width
            let scale = Float(maxDimension) / Float(wBar)
            // Apply that scale to both dimensions to maintain aspect ratio
            hBar = Int(round(Float(hBar) * scale / Float(factor))) * factor
            wBar = Int(round(Float(wBar) * scale / Float(factor))) * factor
        }

        // Ensure dimensions are divisible by the factor
        hBar = (hBar / factor) * factor
        wBar = (wBar / factor) * factor

        // Final sanity check
        if hBar <= 0 || wBar <= 0 {
            throw VLMError.imageProcessingFailure(
                "Invalid target dimensions: \(wBar) × \(hBar)")
        }

        return (hBar, wBar)
    }

    static func replacePaddingTokens(
        in promptTokens: [Int], frames: [THW], paddingToken: String, mergeSize: Int,
        tokenizer: any Tokenizer
    ) throws -> [Int] {
        // Replace single padding token with correct number for each image or video frame
        let placeholderTokens = try tokenizer.encode(
            text: "<|vision_start|>\(paddingToken)<|vision_end|>")
        let placeholderRanges = promptTokens.ranges(of: placeholderTokens)
        guard placeholderRanges.count == frames.count else {
            throw VLMError.processing(
                "Number of placeholder tokens does not match number of frames")
        }
        let mergeLength = mergeSize * mergeSize
        let replacementSequences = try frames.map { frame in
            let paddingCount = frame.product / mergeLength
            return try tokenizer.encode(
                text:
                    "<|vision_start|>\(Array(repeating: paddingToken, count: paddingCount).joined())<|vision_end|>"
            )
        }
        // Build the final array
        var result: [Int] = []
        var currentIndex = promptTokens.startIndex
        for (range, replacement) in zip(placeholderRanges, replacementSequences) {
            // Add tokens before the placeholder
            result.append(contentsOf: promptTokens[currentIndex ..< range.lowerBound])
            // Add replacement sequence
            result.append(contentsOf: replacement)
            currentIndex = range.upperBound
        }
        // Add any remaining tokens after the last replacement
        if currentIndex < promptTokens.endIndex {
            result.append(contentsOf: promptTokens[currentIndex...])
        }
        return result
    }
}
