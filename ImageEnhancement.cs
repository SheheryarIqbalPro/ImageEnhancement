using System;
using System.Collections.Generic;
using System.Drawing;
using System.Drawing.Drawing2D;

public class ReplayBuffer
{
    private List<(float[,], int, float, float[,], bool)> _buffer;
    private int _maxSize;
    private int _currentIndex;

    public ReplayBuffer(int maxSize)
    {
        _maxSize = maxSize;
        _buffer = new List<(float[,], int, float, float[,], bool)>(maxSize);
        _currentIndex = 0;
    }

    public void AddExperience(float[,] state, int action, float reward, float[,] nextState, bool done)
    {
        if (_buffer.Count < _maxSize)
        {
            _buffer.Add((state, action, reward, nextState, done));
        }
        else
        {
            _buffer[_currentIndex] = (state, action, reward, nextState, done);
            _currentIndex = (_currentIndex + 1) % _maxSize;
        }
    }

    public (float[,], int, float, float[,], bool) GetSample(int batchSize)
    {
        var indices = new List<int>();
        var samples = new List<(float[,], int, float, float[,], bool)>(batchSize);
        var rand = new Random();

        while (indices.Count < batchSize)
        {
            var index = rand.Next(_buffer.Count);
            if (!indices.Contains(index))
            {
                indices.Add(index);
                samples.Add(_buffer[index]);
            }
        }

        return samples[batchSize - 1];
    }
}

public class CustomDuelingNeuralNetwork
{
    private int _inputSize;
    private int _outputSize;
    private int _hiddenSize;
    private int _advantageHiddenSize;
    private int _valueHiddenSize;
    private float[,] _weights1;
    private float[,] _weights2;
    private float[,] _advantageWeights1;
    private float[,] _advantageWeights2;
    private float[,] _valueWeights1;
    private float[,] _valueWeights2;

    public CustomDuelingNeuralNetwork(int inputSize, int outputSize, int hiddenSize, int advantageHiddenSize, int valueHiddenSize)
    {
        _inputSize = inputSize;
        _outputSize = outputSize;
        _hiddenSize = hiddenSize;
        _advantageHiddenSize = advantageHiddenSize;
        _valueHiddenSize = valueHiddenSize;

        _weights1 = InitializeWeights(_inputSize, _hiddenSize);
        _weights2 = InitializeWeights(_hiddenSize, _outputSize);
        _advantageWeights1 = InitializeWeights(_hiddenSize, _advantageHiddenSize);
        _advantageWeights2 = InitializeWeights(_advantageHiddenSize, _outputSize);
        _valueWeights1 = InitializeWeights(_hiddenSize, _valueHiddenSize);
        _valueWeights2 = InitializeWeights(_valueHiddenSize, 1);
    }

    public float[,] Forward(float[,] input)
    {
        var hidden = Matrix.Dot(input, _weights1);
        hidden = Matrix.ReLU(hidden);
        var advantageHidden = Matrix.Dot(hidden, _advantageWeights1);
        advantageHidden = Matrix.ReLU(advantageHidden);
        var valueHidden = Matrix.Dot(hidden, _valueWeights1);
        valueHidden = Matrix.ReLU(valueHidden);
        var advantage = Matrix.Dot(advantageHidden, _advantageWeights2);
        var value = Matrix.Dot(valueHidden, _valueWeights2);
        var meanAdvantage = Matrix.Mean(advantage, 1);
        var expandedMeanAdvantage = Matrix.Expand(meanAdvantage, _outputSize);

        var qValues = expandedMeanAdvantage + advantage - Matrix.Expand(value, _outputSize);
        return qValues;
    }

    public void Train(float[,] input, float[,] target, float learningRate)
    {
        var hidden = Matrix.Dot(input, _weights1);
        hidden = Matrix.ReLU(hidden);
        var advantageHidden = Matrix.Dot(hidden, _advantageWeights1);
        advantageHidden = Matrix.ReLU(advantageHidden);
        var valueHidden = Matrix.Dot(hidden, _valueWeights1);
        valueHidden = Matrix.ReLU(valueHidden);
        var advantage = Matrix.Dot(advantageHidden, _advantageWeights2);
        var value = Matrix.Dot(valueHidden, _valueWeights2);
        var meanAdvantage = Matrix.Mean(advantage, 1);
        var expandedMeanAdvantage = Matrix.Expand(meanAdvantage, _outputSize);
        var qValues = expandedMeanAdvantage + advantage - Matrix.Expand(value, _outputSize);
        var error = target - qValues;
        var advantageGradient = Matrix.Dot(hidden.Transpose(), error);
        var valueGradient = Matrix.Dot(hidden.Transpose(), error);
        var advantageError = Matrix.Dot(error, _advantageWeights2.Transpose());
        var advantageHiddenError = Matrix.ReLUGradient(advantageHidden) * advantageError;
        var inputGradient = Matrix.Dot(input.Transpose(), Matrix.Dot(advantageHiddenError, _advantageWeights1.Transpose())) +
                            Matrix.Dot(input.Transpose(), Matrix.Dot(valueGradient, _valueWeights1.Transpose()));
        _advantageWeights1 += learningRate * Matrix.Dot(hidden.Transpose(), advantageHiddenError);
        _advantageWeights2 += learningRate * advantageGradient;
        _valueWeights1 += learningRate * Matrix.Dot(hidden.Transpose(), Matrix.ReLUGradient(valueHidden) * valueGradient);
        _valueWeights2 += learningRate * valueGradient;
        _weights1 += learningRate * inputGradient;
        _weights2 += learningRate * Matrix.Dot(hidden.Transpose(), Matrix.ReLUGradient(advantageHidden) * Matrix.Dot(error, _advantageWeights2.Transpose()));
    }

    private float[,] InitializeWeights(int rows, int columns)
    {
        var rand = new Random();
        var weights = new float[rows, columns];
        for (var i = 0; i < rows; i++)
        {
            for (var j = 0; j < columns; j++)
            {
                weights[i, j] = (float)(rand.NextDouble() * 2 - 1);
            }
        }
        return weights;
    }
}
public class ImageEnhancementEnvironment
{
    private float[,] _originalImage;
    private float[,] _blurredImage;
    private int _maxSteps;
    private int _currentStep;
    public ImageEnhancementEnvironment(float[,] originalImage, float[,] blurredImage, int maxSteps)
    {
        _originalImage = originalImage;
        _blurredImage = blurredImage;
        _maxSteps = maxSteps;
        _currentStep = 0;
    }

    public float[,] Reset()
    {
        _currentStep = 0;
        return _blurredImage;
    }

    public (float[,], float, bool) Step(int action, float[,] qValues)
    {
        var done = IsDone();
        var reward = CalculateReward(action, qValues);

        if (!done)
        {
            var newState = ApplyAction(action);
            _currentStep++;
            done = IsDone();
            return (newState, reward, done);
        }
        else
        {
            return (_blurredImage, reward, true);
        }
    }

    public bool IsDone()
    {
        return _currentStep >= _maxSteps;
    }

    private float[,] ApplyAction(int action)
    {
        var kernelSize = 3;
        var stride = 1;
        var padding = 1;
        var kernel = new float[kernelSize, kernelSize];

        switch (action)
        {
            case 0: // Sharpen
                kernel = new float[,] {
                { -1, -1, -1 },
                { -1,  9, -1 },
                { -1, -1, -1 }
            };
                break;
            case 1: // Edge detection
                kernel = new float[,] {
                { 0, -1,  0 },
                { -1, 4, -1 },
                { 0, -1,  0 }
            };
                break;
            case 2: // Box blur
                kernel = new float[,] {
                { 1, 1, 1 },
                { 1, 1, 1 },
                { 1, 1, 1 }
            };
                break;
            case 3: // Gaussian blur
                kernel = new float[,] {
                { 1, 2, 1 },
                { 2, 4, 2 },
                { 1, 2, 1 }
            };
                break;
            default:
                break;
        }

        var convolved = ImageFiltering.Convolve(_blurredImage, kernel, stride, padding);
        return convolved;
    }

    private float CalculateReward(int action, float[,] qValues)
    {
        var maxQValue = Matrix.Max(qValues);
        var reward = maxQValue;

        switch (action)
        {
            case 0: // Sharpen
                break;
            case 1: // Edge detection
                break;
            case 2: // Box blur
                reward *= 0.9f;
                break;
            case 3: // Gaussian blur
                reward *= 0.8f;
                break;
            default:
                break;
        }

        return reward;
    }
}
public class ImageEnhancementDeblurring
{
    private int _kernelSize;
    private float[,] _kernel;
    public ImageEnhancementDeblurring(int kernelSize)
    {
        _kernelSize = kernelSize;
        _kernel = new float[kernelSize, kernelSize];
        InitializeKernel();
    }

    public float[,] Deblur(float[,] image)
    {
        var stride = 1;
        var padding = (_kernelSize - 1) / 2;
        var deblurred = ImageFiltering.Convolve(image, _kernel, stride, padding);
        return deblurred;
    }

    private void InitializeKernel()
    {
        var sigma = _kernelSize / 5.0f;
        var sum = 0.0f;

        for (var i = 0; i < _kernelSize; i++)
        {
            for (var j = 0; j < _kernelSize; j++)
            {
                var x = i - _kernelSize / 2;
                var y = j - _kernelSize / 2;
                _kernel[i, j] = (float)(Math.Exp(-(x * x + y * y) / (2 * sigma * sigma)));
                sum += _kernel[i, j];
            }
        }

        for (var i = 0; i < _kernelSize; i++)
        {
            for (var j = 0; j < _kernelSize; j++)
            {
                _kernel[i, j] /= sum;
            }
        }
    }
}
public class Matrix
{
    public static float[,] Dot(float[,] a, float[,] b)
    {
        var aRows = a.GetLength(0);
        var aColumns = a.GetLength(1);
        var bRows = b.GetLength(0);
        var bColumns = b.GetLength(1);

        if (aColumns != bRows)
        {
            throw new Exception("Matrix dimensions don't match: " + aColumns + " != " + bRows);
        }

        var result = new float[aRows, bColumns];

        for (var i = 0; i < aRows; i++)
        {
            for (var j = 0; j < bColumns; j++)
            {
                var sum = 0.0f;
                for (var k = 0; k < aColumns; k++)
                {
                    sum += a[i, k] * b[k, j];
                }
                result[i, j] = sum;
            }
        }

        return result;
    }

    public static float[,] ReLU(float[,] a)
    {
        var rows = a.GetLength(0);
        var columns = a.GetLength(1);
        var result = new float[rows, columns];

        for (var i = 0; i < rows; i++)
        {
            for (var j = 0; j < columns; j++)
            {
                result[i, j] = Math.Max(0.0f, a[i, j]);
            }
        }

        return result;
    }

    public static float[,] ReLUGradient(float[,] a)
    {
        var rows = a.GetLength(0);
        var columns = a.GetLength(1);
        var result = new float[rows, columns];

        for (var i = 0; i < rows; i++)
        {
            for (var j = 0; j < columns; j++)
            {
                result[i, j] = a[i, j] > 0.0f ? 1.0f : 0.0f;
            }
        }

        return result;
    }

    public static float[,] Mean(float[,] a, int axis)
    {
        var rows = a.GetLength(0);
        var columns = a.GetLength(1);

        if (axis == 0)
        {
            var result = new float[1, columns];
            for (var j = 0; j < columns; j++)
            {
                var sum = 0.0f;
                for (var i = 0; i < rows; i++)
                {
                    sum += a[i, j];
                }
                result[0, j] = sum / rows;
            }
            return result;
        }
        else if (axis == 1)
        {
            var result = new float[rows, 1];
            for (var i = 0; i < rows; i++)
            {
                var sum = 0.0f;
                for (var j = 0; j < columns; j++)
                {
                    sum += a[i, j];
                }
                result[i, 0] = sum / columns;
            }
            return result;
        }
        else
        {
            throw new Exception("Invalid axis: " + axis);
        }
    }

    public static float[,] Expand(float[,] a, int size)
    {
        var rows = a.GetLength(0);
        var columns = a.GetLength(1);
        var result = new float[rows, size];

        for (var i = 0; i < rows; i++)
        {
            for (var j = 0; j < size; j++)
            {
                result[i, j] = a[i, j % columns];


            }
        }

        return result;
    }

    public static float Max(float[,] a)
    {
        var rows = a.GetLength(0);
        var columns = a.GetLength(1);
        var max = float.MinValue;

        for (var i = 0; i < rows; i++)
        {
            for (var j = 0; j < columns; j++)
            {
                if (a[i, j] > max)
                {
                    max = a[i, j];
                }
            }
        }

        return max;
    }

    public static float[,] Convolve(float[,] input, float[,] kernel, int stride, int padding)
    {
        var inputRows = input.GetLength(0);
        var inputColumns = input.GetLength(1);
        var kernelRows = kernel.GetLength(0);
        var kernelColumns = kernel.GetLength(1);
        var outputRows = (int)Math.Floor((inputRows - kernelRows + 2 * padding) / (float)stride) + 1;
        var outputColumns = (int)Math.Floor((inputColumns - kernelColumns + 2 * padding) / (float)stride) + 1;
        var output = new float[outputRows, outputColumns];

        for (var i = 0; i < outputRows; i++)
        {
            for (var j = 0; j < outputColumns; j++)
            {
                var sum = 0.0f;
                for (var k = 0; k < kernelRows; k++)
                {
                    for (var l = 0; l < kernelColumns; l++)
                    {
                        var rowIndex = i * stride + k - padding;
                        var columnIndex = j * stride + l - padding;
                        if (rowIndex >= 0 && rowIndex < inputRows && columnIndex >= 0 && columnIndex < inputColumns)
                        {
                            sum += input[rowIndex, columnIndex] * kernel[k, l];
                        }
                    }
                }
                output[i, j] = sum;
            }
        }

        return output;
    }
}

public class ReplayBuffer
{
    private int _bufferSize;
    private Queue<(float[,], int, float, float[,], bool)> _buffer;
    public ReplayBuffer(int bufferSize)
    {
        _bufferSize = bufferSize;
        _buffer = new Queue<(float[,], int, float, float[,], bool)>();
    }

    public void AddExperience(float[,] state, int action, float reward, float[,] nextState, bool done)
    {
        if (_buffer.Count == _bufferSize)
        {
            _buffer.Dequeue();
        }
        _buffer.Enqueue((state, action, reward, nextState, done));
    }

    public (float[,], int, float, float[,], bool) Sample(int batchSize)
    {
        var states = new float[batchSize, _buffer.Peek().Item1.GetLength(1)];
        var actions = new int[batchSize];
        var rewards = new float[batchSize];
        var nextStates = new float[batchSize, _buffer.Peek().Item4.GetLength(1)];
        var dones = new bool[batchSize];

        for (var i = 0; i < batchSize; i++)
        {
            var experience = _buffer.ElementAt(new Random().Next(0, _buffer.Count));
            states[i, 0..^1] = experience.Item1.Cast<float>().ToArray();
            actions[i] = experience.Item2;
            rewards[i] = experience.Item3;
            nextStates[i, 0..^1] = experience.Item4.Cast<float>().ToArray();
            dones[i] = experience.Item5;
        }

        return (states, actions, rewards, nextStates, dones);
    }

    public bool IsReady(int batchSize)
    {
        return _buffer.Count >= batchSize;
    }
}

public class CustomDuelingNeuralNetwork
{
    private int _inputSize;
    private int _outputSize;
    private int[] _hiddenLayerSizes;
    private float[,] _inputWeights;
    private float[,] _inputBiases;
    private float[] _hiddenActivation;
    private float[] _hiddenGradient;
    private float[,][] _hiddenWeights;
    private float[,][] _hiddenBiases;
    private float[] _outputActivation;
    private float[] _outputGradient;
    private float[,] _outputWeights;
    private float[,] _outputBiases;
    public CustomDuelingNeuralNetwork(int inputSize = 84 * 84, int outputSize = 4, int[] hiddenLayerSizes = null)
    {
        _inputSize = inputSize;
        _outputSize = outputSize;
        _hiddenLayerSizes = hiddenLayerSizes ?? new int[] { 256, 128 };

        Initialize();
    }

    public float[,] Forward(float[,] input)
    {
        var hidden = Matrix.Dot(input, _inputWeights);
        hidden = Matrix.Expand(hidden, 1);
        hidden = Matrix.ReLU(hidden);
        var hiddenActivations = new float[_hiddenLayerSizes.Length][];

        for (var i = 0; i < _hiddenLayerSizes.Length; i++)
        {
            hiddenActivations[i] = new float[_hiddenLayerSizes[i]];
            hidden = Matrix.Dot(hidden, _hiddenWeights[i]);
            hidden = Matrix.Expand(hidden, 1);
            hidden = Matrix.Add(hidden, _hiddenBiases[i]);
            hidden = Matrix.ReLU(hidden);
        }

        _hiddenActivation = hiddenActivations[^1];
        _outputActivation = Matrix.Dot(hidden, _outputWeights);
        _outputActivation = Matrix.Add(_outputActivation, _outputBiases);
        return _outputActivation;
    }

    public void Backward(float[,] input, float[,] qTarget, float learningRate)
    {
        _outputGradient = (qTarget - _outputActivation);
        var hiddenGradients = new float[_hiddenLayerSizes.Length][];

        for (var i = _hiddenLayerSizes.Length - 1; i >= 0; i--)
        {
            if (i == _hiddenLayerSizes.Length - 1)
            {
                _hiddenGradient = Matrix.Dot(_outputGradient, Matrix.Transpose(_outputWeights));
            }
            else
            {
                _hiddenGradient = Matrix.Dot(_hiddenGradient, Matrix.Transpose(_hiddenWeights[i + 1]));
            }

            hiddenGradients[i] = _hiddenGradient;
            _hiddenGradient *= Matrix.ReLUGradient(Matrix.Expand(_hiddenActivation, _hiddenLayerSizes[i]));
            _hiddenGradient = Matrix.Mean(_hiddenGradient, 1);
            _hiddenGradient = Matrix.Expand(_hiddenGradient, _hiddenLayerSizes[i - 1]);

            if (i == _hiddenLayerSizes.Length - 1)
            {
                _outputWeights += learningRate * Matrix.Dot(Matrix.Transpose(Matrix.Expand(_hiddenActivation, _outputSize)), _outputGradient);
                _outputBiases += learningRate * _outputGradient;
            }
            else
            {
                _hiddenWeights[i] += learningRate * Matrix.Dot(Matrix.Transpose(Matrix.Expand(hiddenGradients[i], _hiddenLayerSizes[i - 1])), Matrix.Expand(hiddenGradients[i - 1], _hiddenLayerSizes[i]));
                _hiddenBiases[i] += learningRate * hiddenGradients[i];
            }
        }

        _inputWeights += learningRate * Matrix.Dot(Matrix.Transpose(input), Matrix.Expand(_hiddenGradient, _inputSize));
        _inputBiases += learningRate * Matrix.Mean(_hiddenGradient, 0);
    }

    private void Initialize()
    {
        _inputWeights = Matrix.RandomUniform(_inputSize, _hiddenLayerSizes[0], -0.01f, 0.01f);
        _inputBiases = Matrix.Zeros(1, _hiddenLayerSizes[0]);
        _hiddenWeights = new float[_hiddenLayerSizes.Length - 1][,][];
        _hiddenBiases = new float[_hiddenLayerSizes.Length - 1][,];

        for (var i = 0; i < _hiddenLayerSizes.Length - 1; i++)
        {
            _hiddenWeights[i] = Matrix.RandomUniform(_hiddenLayerSizes[i], _hiddenLayerSizes[i + 1], -0.01f, 0.01f);
            _hiddenBiases[i] = Matrix.Zeros(1, _hiddenLayerSizes[i + 1]);
        }

        _outputWeights = Matrix.RandomUniform(_hiddenLayerSizes[^1], _outputSize, -0.01f, 0.01f);
        _outputBiases = Matrix.Zeros(1, _outputSize);
    }
}

public class ImageEnhancementDeblurring
{
    private CustomDuelingNeuralNetwork _duelingNetwork;
    private int _inputSize;
    private int _outputSize;
    public ImageEnhancementDeblurring(int inputSize = 84 * 84, int outputSize = 3, int[] hiddenLayerSizes = null)
    {
        _inputSize = inputSize;
        _outputSize = outputSize;
        _duelingNetwork = new CustomDuelingNeuralNetwork(inputSize, outputSize, hiddenLayerSizes);
    }

    public float[,] Enhance(float[,] image)
    {
        return _duelingNetwork.Forward(image);
    }

    public void Train(float[,] state, int action, float reward, float[,] nextState, bool done, float learningRate)
    {
        var qValues = _duelingNetwork.Forward(state);
        var qValuesNext = _duelingNetwork.Forward(nextState);
        var qValueTarget = qValues.Clone() as float[,];

        qValueTarget[0, action] = reward + (done ? 0 : 0.99f * qValuesNext.Max());

        _duelingNetwork.Backward(state, qValueTarget, learningRate);
    }
}

public class ImageEnhancementEnvironment
{
    private ImageEnhancementDeblurring _deblurringModel;
    private Bitmap _image;
    private float[,] _grayscaleImage;
    private int _blurRadius;
    private float _sigma;
    public ImageEnhancementEnvironment(Bitmap image, int blurRadius, float sigma, int inputSize = 84)
    {
        _deblurringModel = new ImageEnhancementDeblurring(inputSize * inputSize * 3, 3, new int[] { 256, 128 });
        _image = image;
        _blurRadius = blurRadius;
        _sigma = sigma;
        _grayscaleImage = ConvertToGrayscale(image);
    }

    public float[,] Reset()
    {
        var x = new Random().Next(0, _image.Width - 84 - 1);
        var y = new Random().Next(0, _image.Height - 84 - 1);
        var state = CropAndNormalize(x, y);

        return state;
    }

    public (float[,] state, float reward, bool done) Step(int action)
    {
        float[,] nextState;
        float reward;

        switch (action)
        {
            case 0:
                nextState = CropAndNormalize(0, 0);
                reward = 0.0f;
                break;
            case 1:
                nextState = CropAndNormalize(1, 0);
                reward = 0.0f;
                break;
            case 2:
                nextState = CropAndNormalize(0, 1);
                reward = 0.0f;
                break;
            case 3:
                nextState = CropAndNormalize(1, 1);
                reward = 0.0f;
                break;
            default:
                throw new ArgumentException("Invalid action");
        }
        var deblurredState = Deblur(nextState);

        var mseBefore = Matrix.Mean(Matrix.Pow(Matrix.Subtract(_grayscaleImage, nextState), 2));
        var mseAfter = Matrix.Mean(Matrix.Pow(Matrix.Subtract(_grayscaleImage, deblurredState), 2));
        var improvement = mseBefore - mseAfter;
        reward += improvement > 0 ? 1 : -1;
        var done = false;

        return (deblurredState, reward, done);
    }

    private float[,] CropAndNormalize(int x, int y)
    {
        var crop = new Bitmap(84, 84);
        var cropGraphics = Graphics.FromImage(crop);
        cropGraphics.DrawImage(_image, new Rectangle(0, 0, 84, 84), new Rectangle(x, y, 84, 84), GraphicsUnit.Pixel);
        var resizedCrop = new Bitmap(crop, new Size(84, 84));
        var normalizedState = Normalize(ConvertToFloat(resizedCrop));

        return normalizedState;
    }

    private float[,] Deblur(float[,] input)
    {
        var kernel = ImageHelper.GaussianKernel(_blurRadius, _sigma);
        var blurredImage = ImageHelper.Convolve(input, kernel);
        var deblurredImage = _deblurringModel.Enhance(blurredImage);
        deblurredImage = ImageHelper.Clamp(deblurredImage, 0, 1);

        return deblurredImage;
    }

    private float[,] ConvertToGrayscale(Bitmap image)
    {
        var grayscale = new float[image.Width, image.Height];

        for (var x = 0; x < image.Width; x++)
        {
            for (var y = 0; y < image.Height; y++)
            {
                var pixel = image.GetPixel(x, y);
                grayscale[x, y] = 0.299f * pixel.R + 0.587f * pixel.G + 0.114f * pixel.B;
            }
        }

        return grayscale;
    }

    private float[,] ConvertToFloat(Bitmap image)
    {
        var floatImage = new float[image.Width, image.Height];

        for (var x = 0; x < image.Width; x++)
        {
            for (var y = 0; y < image.Height; y++)
            {
                var pixel = image.GetPixel(x, y);
                floatImage[x, y] = pixel.R / 255.0f;
                floatImage[x, y + image.Height] = pixel.G / 255.0f;
                floatImage[x, y + 2 * image.Height] = pixel.B / 255.0f;
            }
        }

        return floatImage;
    }

    private float[,] Normalize(float[,] image)
    {
        var mean = Matrix.Mean(image);
        var std = Matrix.StandardDeviation(image);

        return Matrix.Divide(Matrix.Subtract(image, mean), std);
    }
}
public static class ImageHelper
{
    public static float[,] Convolve(float[,] input, float[,] kernel)
    {
        var kernelSize = kernel.GetLength(0);
        var padding = kernelSize / 2;
        var paddedInput = Matrix.Pad(input, padding);
        var outputWidth = input.GetLength(0);
        var outputHeight = input.GetLength(1);
        var output = new float[outputWidth, outputHeight];

        for (var x = 0; x < outputWidth; x++)
        {
            for (var y = 0; y < outputHeight; y++)
            {
                var patch = Matrix.Slice(paddedInput, x, y, kernelSize, kernelSize);
                var convolved = Matrix.Multiply(patch, kernel);
                var sum = Matrix.Sum(convolved);
                output[x, y] = sum;
            }
        }

        return output;
    }

    public static float[,] GaussianKernel(int size, float sigma)
    {
        var kernel = new float[size, size];
        var center = size / 2;
        var sum = 0.0f;

        for (var x = 0; x < size; x++)
        {
            for (var y = 0; y < size; y++)
            {
                var xDistance = x - center;
                var yDistance = y - center;
                kernel[x, y] = (float)Math.Exp(-(xDistance * xDistance + yDistance * yDistance) / (2.0 * sigma * sigma));
                sum += kernel[x, y];
            }
        }

        for (var x = 0; x < size; x++)
        {
            for (var y = 0; y < size; y++)
            {
                kernel[x, y] /= sum;
            }
        }

        return kernel;
    }

    public static float[,] Clamp(float[,] input, float min, float max)
    {
        var output = new float[input.GetLength(0), input.GetLength(1)];

        for (var x = 0; x < input.GetLength(0); x++)
        {
            for (var y = 0; y < input.GetLength(1); y++)
            {
                output[x, y] = Math.Max(Math.Min(input[x, y], max), min);
            }
        }

        return output;
    }
}

public static class Matrix
{
    public static float[,] Zeros(int width, int height)
    {
        return new float[width, height];
    }
    public static float[,] RandomUniform(int width, int height, float min, float max)
    {
        var range = max - min;
        var random = new Random();
        var matrix = new float[width, height];

        for (var x = 0; x < width; x++)
        {
            for (var y = 0; y < height; y++)
            {
                matrix[x, y] = (float)(random.NextDouble() * range + min);
            }
        }

        return matrix;
    }

    public static float[,] Pad(float[,] input, int padding)
    {
        var paddedWidth = input.GetLength(0) + 2 * padding;
        var paddedHeight = input.GetLength(1) + 2 * padding;
        var padded = Zeros(paddedWidth, paddedHeight);

        for (var x = padding; x < paddedWidth - padding; x++)
        {
            for (var y = padding; y < paddedHeight - padding; y++)
            {
                padded[x, y] = input[x - padding, y - padding];
            }
        }

        return padded;
    }

    public static float[,] Slice(float[,] input, int x, int y, int width, int height)
    {
        var slice = Zeros(width, height);
        for (var i = 0; i < width; i++)
        {
            for (var j = 0; j < height; j++)
            {
                slice[i, j] = input[x + i, y + j];
            }
        }

        return slice;
    }

    public static float[,] Add(float[,] a, float[,] b)
    {
        var result = new float[a.GetLength(0), a.GetLength(1)];

        for (var x = 0; x < a.GetLength(0); x++)
        {
            for (var y = 0; y < a.GetLength(1); y++)
            {
                result[x, y] = a[x, y] + b[x, y];
            }
        }

        return result;
    }

    public static float[,] Subtract(float[,] a, float[,] b)
    {
        var result = new float[a.GetLength(0), a.GetLength(1)];

        for (var x = 0; x < a.GetLength(0); x++)
        {
            for (var y = 0; y < a.GetLength(1); y++)
            {
                result[x, y] = a[x, y] - b[x, y];
            }
        }

        return result;
    }

    public static float[,] Multiply(float[,] a, float[,] b)
    {
        var result = new float[a.GetLength(0), a.GetLength(1)];

        for (var x = 0; x < a.GetLength(0); x++)
        {
            for (var y = 0; y < a.GetLength(1); y++)
            {
                result[x, y] = a[x, y] * b[x, y];
            }
        }

        return result;
    }

    public static float Sum(float[,] a)
    {
        var sum = 0.0f;

        for (var x = 0; x < a.GetLength(0); x++)
        {
            for (var y = 0; y < a.GetLength(1); y++)
            {
                sum += a[x, y];
            }
        }

        return sum;
    }

    public static float Mean(float[,] a)
    {
        var sum = Sum(a);

        return sum / (a.GetLength(0) * a.GetLength(1));
    }

    public static float StandardDeviation(float[,] a)
    {
        var mean = Mean(a);
        var variance = 0.0f;

        for (var x = 0; x < a.GetLength(0); x++)
        {
            for (var y = 0; y < a.GetLength(1); y++)
            {
                variance += (a[x, y] - mean) * (a[x, y] - mean);
            }
        }

        variance /= (a.GetLength(0) * a.GetLength(1));

        return (float)Math.Sqrt(variance);
    }

    public static float[,] Pow(float[,] a, float power)
    {
        var result = new float[a.GetLength(0), a.GetLength(1)];

        for (var x = 0; x < a.GetLength(0); x++)
        {
            for (var y = 0; y < a.GetLength(1); y++)
            {
                result[x, y] = (float)Math.Pow(a[x, y], power);
            }
        }

        return result;
    }

    public static float[,] Divide(float[,] a, float b)
    {
        var result = new float[a.GetLength(0), a.GetLength(1)];

        for (var x = 0; x < a.GetLength(0); x++)
        {
            for (var y = 0; y < a.GetLength(1); y++)
            {
                result[x, y] = a[x, y] / b;
            }
        }
        return result;
    }

    public static float Max(float[,] a)
    {
        var max = float.MinValue;

        for (var x = 0; x < a.GetLength(0); x++)
        {
            for (var y = 0; y < a.GetLength(1); y++)
            {
                if (a[x, y] > max)
                {
                    max = a[x, y];
                }
            }
        }

        return max;
    }

    public static float Min(float[,] a)
    {
        var min = float.MaxValue;

        for (var x = 0; x < a.GetLength(0); x++)
        {
            for (var y = 0; y < a.GetLength(1); y++)
            {
                if (a[x, y] < min)
                {
                    min = a[x, y];
                }
            }
        }

        return min;
    }

    public static float[,] Transpose(float[,] a)
    {
        var result = new float[a.GetLength(1), a.GetLength(0)];

        for (var x = 0; x < a.GetLength(0); x++)
        {
            for (var y = 0; y < a.GetLength(1); y++)
            {
                result[y, x] = a[x, y];
            }
        }

        return result;
    }
}

public class ReplayBuffer
{
    private readonly List<(float[,], int, float, float[,], bool)> _buffer = new List<(float[,], int, float, float[,], bool)>();
    private readonly int _bufferSize;
    public ReplayBuffer(int bufferSize)
    {
        _bufferSize = bufferSize;
    }

    public void Add(float[,] state, int action, float reward, float[,] nextState, bool done)
    {
        if (_buffer.Count == _bufferSize)
        {
            _buffer.RemoveAt(0);
        }

        _buffer.Add((state, action, reward, nextState, done));
    }

    public (float[,], int, float, float[,], bool) Sample(int batchSize)
    {
        var samples = new List<(float[,], int, float, float[,], bool)>();

        while (samples.Count < batchSize)
        {
            var index = new Random().Next(_buffer.Count);
            samples.Add(_buffer[index]);
        }

        var states = new float[batchSize, 84, 84, 3];
        var actions = new int[batchSize];
        var rewards = new float[batchSize];
        var nextStates = new float[batchSize, 84, 84, 3];
        var dones = new bool[batchSize];

        for (var i = 0; i < batchSize; i++)
        {
            states[i] = ToTensor(samples[i].Item1);
            actions[i] = samples[i].Item2;
            rewards[i] = samples[i].Item3;
            nextStates[i] = ToTensor(samples[i].Item4);
            dones[i] = samples[i].Item5;
        }

        return (states, actions, rewards, nextStates, dones);
    }

    private float[,,,] ToTensor(float[,] array)
    {
        var width = array.GetLength(0);
        var height = array.GetLength(1);
        var tensor = new float[width, height, 3];

        for (var x = 0; x < width; x++)
        {
            for (var y = 0; y < height; y++)
            {
                tensor[x, y, 0] = array[x, y];
                tensor[x, y, 1] = array[x, y];
                tensor[x, y, 2] = array[x, y];
            }
        }
        return tensor;
    }

    private float[,] FromTensor(float[,,,] tensor)
    {
        var width = tensor.GetLength(0);
        var height = tensor.GetLength(1);
        var array = new float[width, height];

        for (var x = 0; x < width; x++)
        {
            for (var y = 0; y < height; y++)
            {
                array[x, y] = tensor[x, y, 0];
            }
        }

        return array;
    }
}

public class CustomDuelingNeuralNetwork
{
    private readonly Sequential _model;
    private readonly int _numActions;
    public CustomDuelingNeuralNetwork(int numActions = 4)
    {
        _numActions = numActions;
        _model = new Sequential();

        // Define your custom Dueling Q-network architecture
        _model.Add(new Conv2D(32, (8, 8), activation: "relu", inputShape: (84, 84, 3), strides: (4, 4)));
        _model.Add(new Conv2D(64, (4, 4), activation: "relu", strides: (2, 2)));
        _model.Add(new Conv2D(64, (3, 3), activation: "relu", strides: (1, 1)));
        _model.Add(new Flatten());

        // Dueling network split into value and advantage streams
        var valueStream = new Sequential();
        valueStream.Add(new Dense(512, activation: "relu"));
        valueStream.Add(new Dense(1, activation: "linear"));

        var advantageStream = new Sequential();
        advantageStream.Add(new Dense(512, activation: "relu"));
        advantageStream.Add(new Dense(numActions, activation: "linear"));

        // Combine value and advantage streams
        _model.Add(new Lambda(input =>
        {
            var value = valueStream.Call(input);
            var advantage = advantageStream.Call(input);
            var meanAdvantage = K.Mean(advantage, axis: 1, keepdims: true);
            return K.Add(meanAdvantage, K.Subtract(advantage, meanAdvantage) - K.Max(K.Subtract(advantage, meanAdvantage), axis: 1, keepdims: true));
        }));
    }

    public void Train(float[,,,] inputs, float[,] targets, float[,] actions, float[,] weights)
    {
        var loss = new Huber();
        _model.TrainOnBatch(inputs, targets, sampleWeight: weights, loss: loss);
    }

    public float[,] Predict(float[,,,] inputs)
    {
        var predictions = _model.Predict(inputs);
        return predictions[0];
    }

    public float[,] PredictActions(float[,,,] inputs)
    {
        var predictions = _model.Predict(inputs);
        var actions = new float[inputs.GetLength(0), _numActions];

        for (var i = 0; i < inputs.GetLength(0); i++)
        {
            var argmax = 0;

            for (var j = 1; j < _numActions; j++)
            {
                if (predictions[i, j] > predictions[i, argmax])
                {
                    argmax = j;
                }
            }

            actions[i, argmax] = 1;
        }

        return actions;
    }

    public float[,] Gradient(float[,,,] inputs, float[,] targets, float[,] actions, float[,] weights)
    {
        var oss = new Huber();
        var tape = K.GradientTape();
        tape.AddLoss(loss.Call(targets, _model.Call(inputs), sampleWeight: weights));
        var gradients = tape.gradient(loss, _model.Variables);
        var actionGradients = Multiply(gradients[gradients.Length - 1].numpy(), actions);
        return actionGradients;
    }

    public void UpdateTargetNetwork(CustomDuelingNeuralNetwork qNetwork, float tau)
    {
        for (var i = 0; i < _model.Variables.Length; i++)
        {
            _model.Variables[i] = (_model.Variables[i] * (1 - tau)) + (qNetwork._model.Variables[i] * tau);
        }
    }
}



