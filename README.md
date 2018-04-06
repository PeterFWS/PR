# PR
# lecture note for Pattern Recgonition
# Content OverView


CHAP1: Introduction
• What is Pattern Recognition?
	▪ Aims of Image Analysis
	▪ Image analysis versus other techniques
	▪ Differences between images and maps
	▪ Definition of image analysis
	▪ GIS objects derived from digital images
	▪ Components of image analysis
• Strategies of image analysis
	▪ Model-based methods 
		○ Level model of model-based image analysis
			§ Image acquisition
			§ Preprocessing
			§ Segmentation
			§ Recognition/Evaluation
		○ Strategies of model-based image analysis
			§ Bottom-up/Top-down/Mixed mode
	▪ Statistical methods
		○ Standard approach
			§ Image acquisition
			§ Image preprocessing
			§ Calculation of image features
			§ Training
			§ Application of statistical model
		○ From images to features
		○ Supervised classification: Training
• Basic approaches of image analysis
	▪ Semi-automated approach
	▪ Automated approach
		○ model-based image analysis
		○ statistical method
	▪ Neuronal Nets
• Challenges
• Overview of lectures


CHAP2: Human Visual System
CHAP3: Image Acquisition and Preprocessing
• Image Acquisition
	▪ Resolution in remote sensing
		○ Spatial resolution
		○ Spectral resolution
		○ Temporal resolution
		○ Radiometric resolution
• Preprocessing
	▪ Image enhancement
		○ point operations
		○ Local filter operation
			§ Linear Filters
				□ Low pass filter
					® Box filter (rectangular filter)
					® „Hard“ low-pass filter
					® Gauss operator
					-> Binomial filter
			§ Non-linear filters
				□ Rank filter
				-> Minimum/Maximum/Median filter
	▪ Image restauration
• Calculation of derivation
	▪ High pass filters: Edge detection
		○ Gradient operator (1st Derivation)
			§ Gradient based edge detection
				□ Prewitt-Operator
				□ Sobel operator
				-> calculate amplitude and direction of gradient
		○ Laplace-Operator (2nd Derivation) (enhance noise)
		-> can be used to do image sharping
	▪ Derivation of Gaussian (smoothing/low pass)
	-> change sigma, scale space can be spanned
		○ LoG operator (Laplace of Gaussian) (Mexican-Hat-Operator)
		-> noise suppressed
		○ DoG operator (Difference of Gaussians)
		-> approximated by the difference of two low-pass filters with Gaussians of different sigma


CHAP4: Scale Space
• Introduction
• Linear Scale Space
	▪ Convolution with Gaussian kernel
• Scale Space Events
	▪ Blobs
• Blob Detection
	▪ 2D, Laplacian of Gaussian (LoG)
	-> Rotationally symmetric blob detector
	▪ Problem?
	-> The stronger the smoothing, the flatter the resulting curve
	    e.g. 1. derivation of a Gaussian curve, For rising sigma the integral of the area under curve becomes smaller
	▪ Normalization of scale
-> SIFT (Search for characteristic scale, coincides with maximum response of blob detector)
	

CHAP5: Segmentation
• Segmentation of points
	▪ Harris Operator
	▪ Scale Invariant Feature Transform (SIFT)
• Segmentation of Edges and Lines
	▪ Gradient and Laplacian Operators
	▪ Edge Thinning and Location Determination
	▪ Contour Tracing
	▪ Hough Transform
• Segmentation of Regions
	▪ Point-based Approaches
		○ Thresholds derived by Histogram Analysis
	▪ Clustering Approaches
		○ K-Means
		○ Mean Shift
	▪ Edge-based Approaches
		○ Watershed
	▪ Region-based Approaches
		○ Region Growing
		○ Split and Merge
	▪ Graph-based Approaches
	▪ Normalized Cuts


CHAP6: Features
• Radiometric features: for pixels or segments
	▪ Densimetric features (probability density functions)
		○ Histogram
		○ Mean
		○ Variance/Covariance matrix
		○ Skewness
		○ Kurtosis
		○ Entropy
		○ Anisotropy
	▪ Texture Features
		○ Co-occurrence matrices
		-> Haralick features
		○ Filter banks
		-> Textons
		○ Fourier transform
	▪ Structural Features
		○ Local histogram of gradient
		○ HOG features
• Geometric Features: only for segments
	▪ Area
	▪ Perimeter
	▪ Form factors
	▪ Central moments
	▪ Polar distance
	▪ Minimum bounding rectangle (MBR)
		○ Fill factor
• Scaling of Features


CHAP7: Models
• Models in image interpretation
• Object model
	▪ Wireframe model
	▪ Boundary Models
	▪ Half-winged edge model
	▪ Sweep model
	▪ Volume models
	▪ Voxel model
	-> Octrees
	▪ Structural Models
		○ Specific models / CAD models
		○ Parametric models
		○ Generic models
			§ Boundary Models!
			§ Constructive-Solid-Geometry , CSG
	▪ others
• Sensor model
	▪ Sensor properties
	▪ Calibration
• Image model
	▪ Representation by aspect graphs


CHAP8: Representation and Application of Knowledge
• Introduction
	▪ Declarative formulation
	▪ Procedural formulation
• Algorithms and Parameters
• Predicate Logic
• Grammars
• Production Systems
• Frames
• Semantic Networks


CHAP9: Overview of statistical Methods
• Statistical methods in pattern recognition and image analysis
• Tasks and solution strategies
• The feature space
• Taxonomy of statistical methods
	▪ According to the image primitives that are classified
		○ Pixel-based classification
		○ Segment-based classification (object-based classification)
	▪ According to the requirements w.r.t. training data
		○ Supervised classification
		○ Unsupervised classification
	▪ According to the classification procedure
		○ Individual classification
		○ Simultaneous classification
	▪ According to the type of the statistical model
		○ Probabilistic approach (MAP, maximum a posteriori)
			§ Generative methods
			§ Discriminative methods
		○ Non-probabilistic approach
		
• 
	
	▪ According to the models used in probabilistic methods
		○ Parametric techniques
		○ Non-parametric techniques
		
• 


CHAP10: Bayesian Classification
(Probabilistic approach & Generative approach)
• Theorem of Bayes
	▪ Meaning of the terms
	▪ Workflow of Bayesian classification
	▪ Training
• Modelling of the likelihood function P(x|C)
	▪ Non-parametric techniques
	(Cross-Validation to choose parameters)
		○ Histograms
		○ Kernel density estimation
		○ Nearest neighbors techniques
	▪ Parametric techniques
		○ Binary features
		○ Discrete features
		○ Continuous features
			§ Normal distribution
			§ Gaussian mixture model
• Modelling of the prior probability P(C)


CHAP11: Probabilistic Discriminative Classifiers
(Probabilistic approach & Discriminative methods)
• Generative vs. discriminative classifiers
	▪ Discriminative classifiers direct modelling of P(C|x)
	▪ Probabilistic discriminative classifiers
		○ Logistic Regression
		○ Generalized linear models
	▪ Non-probabilistic discriminative classifiers
• Linear Discriminant Function
	▪ Overfitting problem
	▪ Regularization
• Logistic Regression
	▪ Logistic sigmoid function
	▪ Decision boundary
	▪ Geometrical interpretation
• Generalized Linear Models (when data not linearly separable)
	▪ Feature space transformations
	▪ Generalized linear models
• Training
• Multi-class Problems


CHAP12: Non-probabilistic Discriminative Classifiers
• Decision trees
	▪ Binary Tree
	▪ CART (Classification and Regression Trees)
• Bootstrapping
• Random forests (Bootstrapping for CART)
• Boosting
	▪ Application: Face detection
• Support vector machines
	▪ Basic idea
	▪ Linearly separable data
	▪ Non-linear case
		○ Separate in higher dimensional feature space
	▪ Accept (a few) misclassification errors
	▪ Choice of hyper-parameters / multi-class problem
• Neural networks


Chapter 13: Convolutional Neural Networks


Chapter 14: Graphical Methods




