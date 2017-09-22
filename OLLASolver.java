/**
 * <!-- globalinfo-start --> 
 * Implements OLLA Worst Violator Solver, written by Gabriella Melki & VK.
 * <!-- globalinfo-end -->
 */

package vcu.edu.datastreamlearning.ollawv;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

import com.github.javacliparser.FloatOption;
import com.github.javacliparser.IntOption;
import com.yahoo.labs.samoa.instances.Instance;
import com.yahoo.labs.samoa.instances.Instances;
import com.yahoo.labs.samoa.instances.InstancesHeader;
import com.yahoo.labs.samoa.instances.Standardize;

import moa.classifiers.AbstractClassifier;
import moa.core.Measurement;
import moa.core.StringUtils;

public class OLLASolver extends AbstractClassifier {
	private static final long serialVersionUID = 7225375378266360793L;
	
	/**
	 * SVM Penalty Parameter C, command line, default value = 1.0
	 */
	public FloatOption cOption = new FloatOption("C", 'c', "SVM Penalty Parameter.", 1.0);
	/**
	 * Gaussian Gamma Parameter, command line, default value = 0.5
	 */
	public FloatOption gOption = new FloatOption("gamma", 'g', "Gaussian RBF Gamma.", 0.5);
	/**
	 * Binary value to choose whether to use the bias term or not, command line, default value = 1
	 */
	public IntOption bOption = new IntOption("betta",'b',"Use bias or not.",1);
	/**
	 * Set the number of epochs, command line, default value = 0.5
	 */
	public FloatOption eOption = new FloatOption("epochs",'e',"Number of epochs.",0.5);
	/**
	 * Whether to use change-detection method, command line, default value = 0
	 */
	public IntOption changeOption = new IntOption("detection",'d',"Use change detection.",0);
	/**
	 * Set whether to print the entire model, default value = 0
	 */
	public IntOption vOption = new IntOption("verbose",'v',"Print entire model.",0);
	/**
	 * Option for margin width, default value 0.1
	 */
	public FloatOption tOption = new FloatOption("tol",'t',"Tolerance for alpha pruning.",0.1);
	/**
	 * Option to standardize data, default = 1
	 */
	public IntOption standardizeOption = new IntOption("standardize",'p',"Standardize data, 0 mean 1 stdv",0);
	/**
	 * Seed value, default = 1
	 */
	public IntOption seedOption = new IntOption("seed",'r',"Seed value.",1);
	
	/**
	 * Data Header Variables
	 */
	protected int num_data;
	protected int currentSize;
	protected int dim;
	protected int num_classes;
	protected int[] classSizes;
	protected int standardize;
	protected PairwiseTrainingState state;
	/**
	 * Logger for debugging.
	 */
	private static Logger log = new Logger();
	/**
	 * Holds SVM Hyper-parameters.
	 */
	private static SVMParameters params = new SVMParameters();
	/**
	 * Cache + Kernel Evaluator
	 */
	protected Cache cache;
	
	/**
	 * Sets options for model and initializes header for data.
	 */
	@Override
	public void setModelContext(InstancesHeader context){
		// Set SVM parameters from command-line
		params.setC(cOption.getValue());
		params.setGamma(gOption.getValue());
		params.setEpochs(eOption.getValue());
		params.setTol(tOption.getValue());
		// Get data header
		this.num_data = context.numInstances();
		this.dim = context.numAttributes();
		this.num_classes = context.numClasses();
		this.standardize = standardizeOption.getValue();
		// Set the state to be null
		this.state = new PairwiseTrainingState();
		// Set cache to be null
		cache = null;
		// for debugging
		if(vOption.getValue() == 1){
			log.printBarrier();
			log.printf(this.getPurposeString());
			log.printBarrier();
			log.printf(this.getModelContextString());
			log.printBarrier();
		}
	}
	
	/**
	 * Initializes the solver + the pairwise models
	 */
	public void intialize(Instances data){
		// Standardize data
		if(standardize == 1){
			Standardize proc = new Standardize();
			data = proc.convertInstances(data);
		}
		
		// set class index + data information
		num_data = data.numInstances();
		dim = data.numAttributes()-1;
		data.setClassIndex(dim);
		currentSize = num_data;
		
		// initialize class indices
		state.classIndices = new ArrayList<List<Integer>>(num_classes);
		for(int c = 0; c < num_classes; c++){
			state.classIndices.add(new ArrayList<Integer>());
		}
		
		// get class sizes & class indices
		classSizes = new int[num_classes];
		for(int sample = 0; sample < num_data; sample++){
			int label = (int) data.get(sample).classValue();
			classSizes[label]++;
			
			state.classIndices.get(label).add(sample);
		}
		
		// initialize pairwise models & assign size of each based on the class sizes
		state.models = new ArrayList<PairwiseTrainingResult>();
		for(int i = 0; i < num_classes; i++){
			for(int j = (i+1); j < num_classes; j++){
				int size = classSizes[i] + classSizes[j];
				Tuple<Integer,Integer> trainPair = new Tuple<Integer,Integer>(i,j);
				state.models.add(new PairwiseTrainingResult(params,size,trainPair));
			}
		}
		
		// set the last label number as max label
		state.setLabelNumber(num_classes);
		
		// build the cache
		cache = new Cache(data,num_data,dim,params);
		
		// for debugging
		if(vOption.getValue() == 1){
			log.printBarrier();
			log.println("intialize(Instances data)::");
			log.printf("\tTotal number of data: %d\n", this.num_data);
			log.printf("\tDimensionality: %d\n", this.dim);
			log.printf("\tNumber of classes: %d\n", this.num_classes);
			log.printf("\tStandardize data: %d\n", this.standardize);
			log.printf("\tSeed: %d\n", this.randomSeed);
			log.printf("\tClass Sizes: %s\n", Arrays.toString(classSizes));
			log.printf("\tCurrent Size: %d\n", currentSize);
			log.printBarrier();
		}
	}
	
	/**
	 * Train on a batch of instances.
	 */
	@Override
	public void trainOnInstances(Instances data) {
		if(cache == null){
			intialize(data);
		}
		// set up the environment for each pairwise model
		int svNum = 0;
		int totalSize = num_data;
		for(int i = 0; i < state.models.size(); i++){
			Tuple<Integer,Integer> trainPair = state.models.get(i).trainingLabels;
			int size = reorderSamples(trainPair, totalSize, data);
			
			// for debugging
			if(vOption.getValue() == 1){
				log.printInstances(data);
				log.printBarrier();
			}
			
			setCurrentSize(size);
			cache.setLabel(trainPair.second);
			
			// TODO: see about resetting cache (if needed)
			
			// train binary OLLAWV model
			cache.trainForCache(data);
			// update pairwise training model
			state.models.get(i).setAlphas(cache.getAlphas());
			state.models.get(i).setInd(cache.getInd());
			state.models.get(i).setBias(cache.bias);
			state.models.get(i).setSvnumber(cache.svnumber);
			// save largest svnumber
			if(state.models.get(i).getSvnumber() > svNum){
				svNum = state.models.get(i).getSvnumber();
			}
			// for debugging
			if(vOption.getValue() == 1){
				log.printBarrier();
				log.print("trainOnInstances(Instances data)::");
				log.printf("\tPair 1:(%d,%d)\n", trainPair.first, trainPair.second);
				log.printf("\tNumber of Support Vectors: %d\n", cache.svnumber);
				log.printf("\tBias: %d\n", cache.bias);
				log.println("   Alphas: "+cache.getAlphas().toString());
				log.println("   Inds: "+cache.getInd().toString());
				log.printBarrier();
			}
		}
		// set the sv number to be the highest of the models
		state.setSvNumber(svNum);
		currentSize = totalSize;
	}
	
	/**
	 * Set current size of the pairwise model
	 */
	public void setCurrentSize(int size){
		currentSize = size;
		cache.setCurrentSize(size);
	}
	
	/**
	 * Orders samples based on the current label training pair
	 */
	public int reorderSamples(Tuple<Integer,Integer> trainPair, int size, Instances data){
		int first = trainPair.first;
		int second = trainPair.second;
		int train = 0;
		int test = size-1;		
		while(train <= test){
			while(train < size && (data.get(train).classValue() == first || data.get(train).classValue() == second)){
				train++;
			}
			while(test >= 0 && (data.get(test).classValue() != first && data.get(test).classValue() != second)){
				test--;
			}
			if(train < test){
				data.swap(train++, test--);
			}
		}
		return train;
	}
	
	/** 
	 * Test on a batch of instances.
	 */
	@Override
	public double[] getVotesForInstance(Instance inst) {
		// TODO Tester method for single instance
		return null;
	}
	
	// TODO add toString method
	// TODO add reset method

	@Override
	public void resetLearningImpl() {
		// TODO reset the learner
		
	}

	@Override
	public void trainOnInstanceImpl(Instance inst) {
		// TODO Trains the classifier on a given instance
		log.print("ERROR: trainOnInstanceImpl was called and no implementation for this yet.\n");
		System.err.println("trainOnInstanceImpl::This sould not happen.");
	}

	@Override
	protected Measurement[] getModelMeasurementsImpl() {
		return null;
	}
	
	/**
	 *  Returns the name of the solver. 
	 */
	@Override
    public String getPurposeString() {
        return "OnLine Learning Algorithm using Worst Violator (OLLAWV), VK and GM (2017).\n";
    }
	
	/**
	 * Prints the model context.
	 */
	public String getModelContextString(){
		StringBuffer buff = new StringBuffer();
		buff.append("Data Standardization: "+standardize+"\n");
		buff.append("SVM Parameter C: "+params.getC()+"\n");
		buff.append("RBF Parameter gamma: "+params.getGamma()+"\n");
		buff.append("Using Bias: "+params.getBetta()+"\n");
		buff.append("Margin Tol: "+params.getTol()+"\n");
		buff.append("Number of Epochs: "+params.getEpochs()+"\n");
		buff.append("Using Change Detection: "+changeOption.getValue()+"\n");
        return buff.toString();
	}
	
	/**
	 *  Prints parameter out to string builder (auto-generated inherited). 
	 */
	@Override
	public void getModelDescription(StringBuilder out, int indent) {
		StringUtils.appendIndented(out, indent, toString());
        StringUtils.appendNewline(out);
	}

	public static Logger getLog() {
		return log;
	}

	public static void setLog(Logger log) {
		OLLASolver.log = log;
	}
	
	@Override
	public boolean isRandomizable() {
		return false;
	}

}
