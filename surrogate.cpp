#include <iostream>
#include <vector>
#include <cmath>
#include <numeric>    
#include <algorithm>  
#include <fstream>     

// Define a small epsilon for numerical stability in matrix inversion/solving
const double EPSILON = 1e-9;

// Structure to hold input features (e.g., Mach number, Angle of Attack)
struct InputFeatures {
    double mach_number;
    double angle_of_attack;

    // Default constructor
    InputFeatures(double mach = 0.0, double aoa = 0.0) : mach_number(mach), angle_of_attack(aoa) {}

    // Function to calculate Euclidean distance between two feature sets
    double distance(const InputFeatures& other) const {
        double dm = mach_number - other.mach_number;
        double dao = angle_of_attack - other.angle_of_attack;
        return std::sqrt(dm * dm + dao * dao);
    }

    // Convert to a vector for processing
    std::vector<double> to_vector() const {
        return {mach_number, angle_of_attack};
    }
};

// Structure to hold output aerodynamic coefficients (e.g., Lift, Drag, Moment)
struct AerodynamicCoefficients {
    double cl; // Lift Coefficient
    double cd; // Drag Coefficient
    double cm; // Moment Coefficient

    // Default constructor
    AerodynamicCoefficients(double lift = 0.0, double drag = 0.0, double moment = 0.0)
        : cl(lift), cd(drag), cm(moment) {}

    // Convert to a vector for processing
    std::vector<double> to_vector() const {
        return {cl, cd, cm};
    }
};

// Represents a single data point for training
struct DataPoint {
    InputFeatures features;
    AerodynamicCoefficients coeffs;

    DataPoint(const InputFeatures& f, const AerodynamicCoefficients& c)
        : features(f), coeffs(c) {}
};

// Gaussian Radial Basis Function kernel
// Phi(r) = exp(-(epsilon * r)^2)
double gaussian_rbf(double r, double epsilon) {
    return std::exp(-(epsilon * r) * (epsilon * r));
}

// A simplified matrix class for demonstration purposes.
// In a real application, you would use a robust linear algebra library like Eigen.
class Matrix {
public:
    std::vector<std::vector<double>> data;
    size_t rows;
    size_t cols;

    Matrix(size_t r, size_t c) : rows(r), cols(c) {
        data.resize(rows, std::vector<double>(cols, 0.0));
    }

    // Access element
    double& operator()(size_t r, size_t c) {
        return data[r][c];
    }

    const double& operator()(size_t r, size_t c) const {
        return data[r][c];
    }

    // Simple matrix-vector multiplication (MxV)
    std::vector<double> multiply(const std::vector<double>& vec) const {
        if (cols != vec.size()) {
            std::cerr << "Error: Matrix-vector dimensions mismatch for multiplication." << std::endl;
            return {};
        }
        std::vector<double> result(rows, 0.0);
        for (size_t i = 0; i < rows; ++i) {
            for (size_t j = 0; j < cols; ++j) {
                result[i] += data[i][j] * vec[j];
            }
        }
        return result;
    }

    // Simple Gaussian elimination for solving Ax = b (for square matrices)
    // This is a basic implementation and not numerically stable for all cases.
    // A robust linear algebra library is highly recommended for real applications.
    std::vector<double> solve(const std::vector<double>& b) const {
        if (rows != cols || rows != b.size()) {
            std::cerr << "Error: Matrix must be square and dimensions must match vector b for solving." << std::endl;
            return {};
        }

        size_t n = rows;
        Matrix A_copy = *this; // Create a copy of the matrix
        std::vector<double> b_copy = b; // Create a copy of the vector

        // Forward elimination
        for (size_t k = 0; k < n; ++k) {
            // Find pivot for column k
            size_t pivot_row = k;
            for (size_t i = k + 1; i < n; ++i) {
                if (std::abs(A_copy(i, k)) > std::abs(A_copy(pivot_row, k))) {
                    pivot_row = i;
                }
            }

            // Swap current row with pivot row
            if (pivot_row != k) {
                std::swap(A_copy.data[k], A_copy.data[pivot_row]);
                std::swap(b_copy[k], b_copy[pivot_row]);
            }

            // Check for singular matrix
            if (std::abs(A_copy(k, k)) < EPSILON) {
                std::cerr << "Error: Matrix is singular or ill-conditioned. Cannot solve." << std::endl;
                return {};
            }

            // Eliminate lower rows
            for (size_t i = k + 1; i < n; ++i) {
                double factor = A_copy(i, k) / A_copy(k, k);
                for (size_t j = k; j < n; ++j) {
                    A_copy(i, j) -= factor * A_copy(k, j);
                }
                b_copy[i] -= factor * b_copy[k];
            }
        }

        // Back substitution
        std::vector<double> x(n);
        for (int i = n - 1; i >= 0; --i) {
            double sum = 0.0;
            for (size_t j = i + 1; j < n; ++j) {
                sum += A_copy(i, j) * x[j];
            }
            x[i] = (b_copy[i] - sum) / A_copy(i, i);
        }
        return x;
    }

    void print() const {
        for (size_t i = 0; i < rows; ++i) {
            for (size_t j = 0; j < cols; ++j) {
                std::cout << data[i][j] << "\t";
            }
            std::cout << std::endl;
        }
    }
};


// RBF Interpolator class
class RBFInterpolator {
public:
    RBFInterpolator(double epsilon = 1.0) : epsilon_(epsilon) {}

    // Train the RBF model using a set of data points
    // This involves solving a linear system to find the weights (lambda)
    void train(const std::vector<DataPoint>& training_data) {
        if (training_data.empty()) {
            std::cerr << "Training data is empty. Cannot train RBF model." << std::endl;
            return;
        }

        training_points_ = training_data;
        size_t n = training_data.size();

        // Construct the interpolation matrix A
        // A_ij = Phi(||x_i - x_j||)
        Matrix A(n, n);
        for (size_t i = 0; i < n; ++i) {
            for (size_t j = 0; j < n; ++j) {
                double r = training_data[i].features.distance(training_data[j].features);
                A(i, j) = gaussian_rbf(r, epsilon_);
            }
        }

        // Construct the right-hand side vectors for Cl, Cd, Cm
        std::vector<double> b_cl(n);
        std::vector<double> b_cd(n);
        std::vector<double> b_cm(n);

        for (size_t i = 0; i < n; ++i) {
            b_cl[i] = training_data[i].coeffs.cl;
            b_cd[i] = training_data[i].coeffs.cd;
            b_cm[i] = training_data[i].coeffs.cm;
        }

        // Solve the linear system A * lambda = b for each coefficient
        // lambda_cl, lambda_cd, lambda_cm are the weights for each RBF
        // In a real application, use a robust linear solver from a library like Eigen.
        std::cout << "Solving linear system for RBF weights..." << std::endl;
        lambda_cl_ = A.solve(b_cl);
        lambda_cd_ = A.solve(b_cd);
        lambda_cm_ = A.solve(b_cm);

        if (lambda_cl_.empty() || lambda_cd_.empty() || lambda_cm_.empty()) {
            std::cerr << "Failed to solve linear system. RBF model might not be trained correctly." << std::endl;
        } else {
            std::cout << "RBF model trained successfully with " << n << " data points." << std::endl;
        }
    }

    // Predict aerodynamic coefficients for new input features
    AerodynamicCoefficients predict(const InputFeatures& new_features) const {
        if (training_points_.empty() || lambda_cl_.empty()) {
            std::cerr << "RBF model not trained. Cannot make predictions." << std::endl;
            return AerodynamicCoefficients(); // Return default
        }

        double predicted_cl = 0.0;
        double predicted_cd = 0.0;
        double predicted_cm = 0.0;

        size_t n = training_points_.size();
        for (size_t i = 0; i < n; ++i) {
            double r = new_features.distance(training_points_[i].features);
            double phi_r = gaussian_rbf(r, epsilon_);

            predicted_cl += lambda_cl_[i] * phi_r;
            predicted_cd += lambda_cd_[i] * phi_r;
            predicted_cm += lambda_cm_[i] * phi_r;
        }

        return AerodynamicCoefficients(predicted_cl, predicted_cd, predicted_cm);
    }

private:
    std::vector<DataPoint> training_points_;
    std::vector<double> lambda_cl_; // Weights for Lift Coefficient
    std::vector<double> lambda_cd_; // Weights for Drag Coefficient
    std::vector<double> lambda_cm_; // Weights for Moment Coefficient
    double epsilon_; // Shape parameter for the Gaussian RBF
};

// --- Data Preprocessing and Dimensionality Reduction (Conceptual PCA) ---
class DataPreprocessor {
public:
    DataPreprocessor() : is_trained_(false) {}

    // Trains the preprocessor (e.g., calculates min/max for normalization, or PCA components)
    void train(const std::vector<InputFeatures>& data) {
        if (data.empty()) {
            std::cerr << "Data for preprocessor training is empty." << std::endl;
            return;
        }

        // For normalization (Min-Max scaling)
        min_values_.resize(data[0].to_vector().size(), std::numeric_limits<double>::max());
        max_values_.resize(data[0].to_vector().size(), std::numeric_limits<double>::lowest());

        for (const auto& features : data) {
            std::vector<double> vec = features.to_vector();
            for (size_t i = 0; i < vec.size(); ++i) {
                min_values_[i] = std::min(min_values_[i], vec[i]);
                max_values_[i] = std::max(max_values_[i], vec[i]);
            }
        }

        // Conceptual PCA training:
        // In a real scenario, you would calculate the covariance matrix, perform
        // eigenvalue decomposition, and store the principal components here.
        // This is highly complex and usually done with a dedicated linear algebra library
        // like Eigen, or by loading pre-computed components from a Python training script.
        std::cout << "DataPreprocessor: Training for normalization completed." << std::endl;
        std::cout << "DataPreprocessor: PCA training would involve covariance matrix calculation and eigenvalue decomposition here." << std::endl;
        is_trained_ = true;
    }

    // Normalizes input features
    InputFeatures normalize(const InputFeatures& features) const {
        if (!is_trained_) {
            std::cerr << "Preprocessor not trained. Cannot normalize." << std::endl;
            return features;
        }
        std::vector<double> vec = features.to_vector();
        InputFeatures normalized_features;
        if (vec.size() != min_values_.size()) {
            std::cerr << "Feature dimension mismatch during normalization." << std::endl;
            return features;
        }

        // Apply Min-Max scaling
        normalized_features.mach_number = (vec[0] - min_values_[0]) / (max_values_[0] - min_values_[0] + EPSILON);
        normalized_features.angle_of_attack = (vec[1] - min_values_[1]) / (max_values_[1] - min_values_[1] + EPSILON);
        return normalized_features;
    }

    // Denormalizes output coefficients (if outputs were normalized during training)
    AerodynamicCoefficients denormalize_coeffs(const AerodynamicCoefficients& coeffs,
                                               const std::vector<double>& min_coeffs,
                                               const std::vector<double>& max_coeffs) const {
        if (min_coeffs.empty() || max_coeffs.empty() || min_coeffs.size() != coeffs.to_vector().size()) {
            std::cerr << "Denormalization parameters missing or mismatched." << std::endl;
            return coeffs;
        }
        AerodynamicCoefficients denormalized_coeffs;
        denormalized_coeffs.cl = coeffs.cl * (max_coeffs[0] - min_coeffs[0]) + min_coeffs[0];
        denormalized_coeffs.cd = coeffs.cd * (max_coeffs[1] - min_coeffs[1]) + min_coeffs[1];
        denormalized_coeffs.cm = coeffs.cm * (max_coeffs[2] - min_coeffs[2]) + min_coeffs[2];
        return denormalized_coeffs;
    }

    // Conceptual PCA transformation
    // This method would take the original features and apply a pre-computed PCA transformation
    // to reduce their dimensionality.
    std::vector<double> apply_pca_transform(const InputFeatures& features) const {
        // In a real scenario, this would involve matrix multiplication of the
        // mean-centered features with the principal components (eigenvectors).
        // For demonstration, we'll just return the original vector.
        std::cout << "DataPreprocessor: Applying conceptual PCA transform (returning original features for demo)." << std::endl;
        return features.to_vector(); // Placeholder
    }

private:
    std::vector<double> min_values_;
    std::vector<double> max_values_;
    bool is_trained_;
};

// --- Conceptual Gaussian Process Regressor ---
// In a real application, GPR implementation involves complex kernel matrix computations,
// Cholesky decomposition for inversion, and hyperparameter optimization.
// It's typically trained in Python (e.g., scikit-learn's GaussianProcessRegressor)
// and then its learned parameters (kernel, alpha, etc.) are used for prediction in C++.
class GaussianProcessRegressor {
public:
    GaussianProcessRegressor() : is_trained_(false) {}

    // Placeholder for training. In reality, this would involve solving
    // for kernel parameters and the inverse of the kernel matrix.
    void train(const std::vector<DataPoint>& training_data) {
        std::cout << "GaussianProcessRegressor: Training is a complex process involving kernel matrix inversion and hyperparameter optimization." << std::endl;
        std::cout << "GaussianProcessRegressor: Typically, this model would be trained in Python and its parameters loaded here." << std::endl;
        // For demonstration, we'll just store the training data
        // and simulate some pre-computed parameters.
        training_points_ = training_data;
        // Simulate some pre-computed alpha values (weights) and kernel parameters
        // These would be loaded from a file after Python training.
        alpha_cl_.resize(training_data.size(), 0.1); // Dummy values
        alpha_cd_.resize(training_data.size(), 0.01); // Dummy values
        alpha_cm_.resize(training_data.size(), -0.05); // Dummy values
        kernel_length_scale_ = 0.5; // Dummy hyperparameter
        kernel_amplitude_ = 1.0;    // Dummy hyperparameter
        is_trained_ = true;
    }

    // Predict using the trained GP model
    // This assumes kernel parameters and alpha (weights) are already known.
    AerodynamicCoefficients predict(const InputFeatures& new_features) const {
        if (!is_trained_) {
            std::cerr << "GaussianProcessRegressor not trained. Cannot make predictions." << std::endl;
            return AerodynamicCoefficients();
        }

        double predicted_cl = 0.0;
        double predicted_cd = 0.0;
        double predicted_cm = 0.0;

        // Calculate the K_star (covariance between test point and training points)
        // using the RBF kernel (or any other chosen kernel for the GP)
        for (size_t i = 0; i < training_points_.size(); ++i) {
            double r = new_features.distance(training_points_[i].features);
            // Use the kernel function with learned hyperparameters
            double k_star_i = kernel_amplitude_ * std::exp(-0.5 * std::pow(r / kernel_length_scale_, 2));

            predicted_cl += k_star_i * alpha_cl_[i];
            predicted_cd += k_star_i * alpha_cd_[i];
            predicted_cm += k_star_i * alpha_cm_[i];
        }

        // In a full GPR, you would also calculate the variance of the prediction.
        return AerodynamicCoefficients(predicted_cl, predicted_cd, predicted_cm);
    }

private:
    std::vector<DataPoint> training_points_; // Stored for kernel calculation during prediction
    std::vector<double> alpha_cl_; // Equivalent to (K + sigma_n^2 * I)^-1 * y_cl
    std::vector<double> alpha_cd_;
    std::vector<double> alpha_cm_;
    double kernel_length_scale_; // Learned hyperparameter
    double kernel_amplitude_;    // Learned hyperparameter
    bool is_trained_;
};


// --- Conceptual Neural Network Class ---
// This class demonstrates how you would load and use a pre-trained NN model
// (e.g., from ONNX or TensorFlow Lite format) for inference.
// Implementing a full NN from scratch with backpropagation is beyond this scope.
class NeuralNetwork {
public:
    NeuralNetwork() : is_model_loaded_(false) {}

    // Conceptual function to load a pre-trained model
    bool load_model(const std::string& model_path) {
        std::cout << "NeuralNetwork: Attempting to load model from " << model_path << std::endl;
        // In a real application, this would use a library like ONNX Runtime C++ API
        // or TensorFlow Lite C++ API to load the model file (.onnx, .tflite).
        // For demonstration, we'll just simulate success.
        std::ifstream file(model_path);
        if (file.good()) {
            std::cout << "NeuralNetwork: Model loaded successfully (conceptual)." << std::endl;
            is_model_loaded_ = true;
            // Initialize dummy weights/biases if not loading a real model
            // For a 2-input, 3-output model (like our features to coeffs) with one hidden layer
            // This is a very simplified placeholder.
            hidden_weights_ = {{0.5, 0.2}, {0.1, -0.3}, {0.4, 0.6}}; // 3 hidden neurons, 2 inputs
            hidden_biases_ = {0.1, 0.2, 0.3};
            output_weights_ = {{1.0, 0.5, -0.2}, {0.3, 1.2, 0.1}, {-0.1, 0.4, 0.9}}; // 3 outputs, 3 hidden neurons
            output_biases_ = {0.05, 0.02, 0.08};
            return true;
        } else {
            std::cerr << "NeuralNetwork: Failed to load model from " << model_path << std::endl;
            return false;
        }
    }

    // Predict aerodynamic coefficients using the loaded NN model
    AerodynamicCoefficients predict(const InputFeatures& new_features) const {
        if (!is_model_loaded_) {
            std::cerr << "NeuralNetwork model not loaded. Cannot make predictions." << std::endl;
            return AerodynamicCoefficients();
        }

        // Convert input features to a vector
        std::vector<double> inputs = new_features.to_vector();

        // --- Forward pass through a simplified NN (conceptual) ---
        // This simulates a single hidden layer with ReLU activation and a linear output layer.
        // In reality, this would be handled by the loaded model's inference engine.

        // Hidden layer calculation (Input * Weights + Biases)
        std::vector<double> hidden_layer_output(hidden_weights_.size(), 0.0);
        for (size_t i = 0; i < hidden_weights_.size(); ++i) { // For each hidden neuron
            for (size_t j = 0; j < inputs.size(); ++j) { // For each input
                hidden_layer_output[i] += inputs[j] * hidden_weights_[i][j];
            }
            hidden_layer_output[i] += hidden_biases_[i];
            // Apply ReLU activation
            hidden_layer_output[i] = std::max(0.0, hidden_layer_output[i]);
        }

        // Output layer calculation (Hidden_Output * Weights + Biases)
        std::vector<double> outputs(output_weights_.size(), 0.0);
        for (size_t i = 0; i < output_weights_.size(); ++i) { // For each output neuron (Cl, Cd, Cm)
            for (size_t j = 0; j < hidden_layer_output.size(); ++j) { // For each hidden neuron output
                outputs[i] += hidden_layer_output[j] * output_weights_[i][j];
            }
            outputs[i] += output_biases_[i];
        }

        return AerodynamicCoefficients(outputs[0], outputs[1], outputs[2]);
    }

private:
    bool is_model_loaded_;
    // Placeholder for NN parameters (weights, biases)
    // In a real scenario, these would be part of the loaded model.
    std::vector<std::vector<double>> hidden_weights_;
    std::vector<double> hidden_biases_;
    std::vector<std::vector<double>> output_weights_;
    std::vector<double> output_biases_;
};


int main() {
    std::cout << "Starting Surrogate Modeling Example for Transonic Aerodynamics with RBF, GP, and NN concepts..." << std::endl;

    // --- 1. Generate Synthetic Training Data ---
    // In a real scenario, this data would come from CFD simulations.
    std::vector<DataPoint> training_data;
    std::vector<InputFeatures> raw_input_features; // For preprocessor training

    for (int i = 0; i <= 10; ++i) {
        double mach = 0.7 + i * 0.02; // Mach from 0.7 to 0.9
        for (int j = 0; j <= 5; ++j) {
            double aoa = 2.0 + j * 0.5; // AoA from 2.0 to 4.5 degrees

            InputFeatures features(mach, aoa);
            raw_input_features.push_back(features); // Store raw features for preprocessor

            // Simulate some non-linear aerodynamic behavior
            double cl = 0.8 + 0.5 * aoa / 5.0 - 2.0 * std::pow(mach - 0.8, 2); // Peak around Mach 0.8
            double cd = 0.02 + 0.01 * aoa / 5.0 + 0.5 * std::pow(mach - 0.85, 4); // Drag rise around Mach 0.85
            double cm = -0.1 - 0.05 * aoa / 5.0 + 0.1 * (mach - 0.7);

            training_data.emplace_back(features, AerodynamicCoefficients(cl, cd, cm));
        }
    }

    std::cout << "Generated " << training_data.size() << " synthetic training data points." << std::endl;

    // --- 2. Data Preprocessing and Dimensionality Reduction (Conceptual) ---
    DataPreprocessor preprocessor;
    preprocessor.train(raw_input_features);

    // Example of normalizing a test feature
    InputFeatures raw_test_feature(0.78, 3.5);
    InputFeatures normalized_test_feature = preprocessor.normalize(raw_test_feature);
    std::cout << "\nRaw Test Feature: Mach=" << raw_test_feature.mach_number
              << ", AoA=" << raw_test_feature.angle_of_attack << std::endl;
    std::cout << "Normalized Test Feature: Mach=" << normalized_test_feature.mach_number
              << ", AoA=" << normalized_test_feature.angle_of_attack << std::endl;

    // Example of applying conceptual PCA
    std::vector<double> pca_transformed_features = preprocessor.apply_pca_transform(normalized_test_feature);
    std::cout << "PCA Transformed Features (conceptual): ";
    for (double val : pca_transformed_features) {
        std::cout << val << " ";
    }
    std::cout << std::endl;


    // --- 3. Initialize and Train the RBF Interpolator ---
    std::cout << "\n--- RBF Interpolator ---" << std::endl;
    RBFInterpolator rbf_model(5.0); // Example epsilon value
    rbf_model.train(training_data);

    InputFeatures test_features_rbf(0.75, 3.0);
    AerodynamicCoefficients predicted_coeffs_rbf = rbf_model.predict(test_features_rbf);
    std::cout << "RBF Prediction for Mach=" << test_features_rbf.mach_number
              << ", AoA=" << test_features_rbf.angle_of_attack << ":\n"
              << "  Cl: " << predicted_coeffs_rbf.cl << ", Cd: " << predicted_coeffs_rbf.cd
              << ", Cm: " << predicted_coeffs_rbf.cm << std::endl;


    // --- 4. Conceptual Gaussian Process Regressor ---
    std::cout << "\n--- Gaussian Process Regressor (Conceptual) ---" << std::endl;
    GaussianProcessRegressor gp_model;
    // In a real scenario, you'd train this in Python and load parameters.
    // For this demo, we'll just "train" it with the data to set up dummy parameters.
    gp_model.train(training_data);

    InputFeatures test_features_gp(0.82, 4.0);
    AerodynamicCoefficients predicted_coeffs_gp = gp_model.predict(test_features_gp);
    std::cout << "GP Prediction for Mach=" << test_features_gp.mach_number
              << ", AoA=" << test_features_gp.angle_of_attack << ":\n"
              << "  Cl: " << predicted_coeffs_gp.cl << ", Cd: " << predicted_coeffs_gp.cd
              << ", Cm: " << predicted_coeffs_gp.cm << std::endl;


    // --- 5. Conceptual Neural Network ---
    std::cout << "\n--- Neural Network (Conceptual) ---" << std::endl;
    NeuralNetwork nn_model;
    // Attempt to load a dummy model file. Create an empty file named "model.bin"
    // in the same directory as your executable for this to "succeed" conceptually.
    if (nn_model.load_model("model.bin")) {
        InputFeatures test_features_nn(0.88, 2.5);
        // If using normalized inputs for NN, pass normalized_test_feature here
        AerodynamicCoefficients predicted_coeffs_nn = nn_model.predict(test_features_nn);
        std::cout << "NN Prediction for Mach=" << test_features_nn.mach_number
                  << ", AoA=" << test_features_nn.angle_of_attack << ":\n"
                  << "  Cl: " << predicted_coeffs_nn.cl << ", Cd: " << predicted_coeffs_nn.cd
                  << ", Cm: " << predicted_coeffs_nn.cm << std::endl;
    } else {
        std::cerr << "NN model not loaded. Skipping NN prediction." << std::endl;
    }

    return 0;
}
