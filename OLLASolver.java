/**
 * <!-- globalinfo-start --> 
 * Implements OLLA Worst Violator Solver, written by Gabriella Melki & VK.
 * <!-- globalinfo-end -->
 */

package vcu.edu.datastreamlearning.ollawv;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.Iterator;

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
	protected int dim;
	protected int num_classes;
	protected int[] classSizes;
	protected int standardize;
	protected PairwiseTrainingState state;
	/**
	 * Logger for debugging.
	 */
	protected static Logger log = new Logger();
	/**
	 * Holds SVM Hyper-parameters.
	 */
	private static SVMParameters params;
	/**
	 * Cache + Kernel Evaluator
	 */
	protected Cache cache;
	protected KernelEvaluator eval;
	
	/**
	 * Sets options for model and initializes header for data.
	 */
	@Override
	public void setModelContext(InstancesHeader context){
		// Set SVM parameters from command-line
		params = new SVMParameters();
		params.setC(cOption.getValue());
		params.setGamma(gOption.getValue());
		params.setEpochs(eOption.getValue());
		params.setTol(tOption.getValue());
		// Get data header
		this.num_data = context.numInstances();
		this.dim = context.numAttributes()-1;
		this.num_classes = context.numClasses();
		this.standardize = standardizeOption.getValue();
		// Set the state to be null
		this.state = new PairwiseTrainingState();
		// Set cache to be null
		cache = null;
		eval = null;
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
		
		// set class index & data information
		num_data = data.numInstances();
		dim = data.numAttributes()-1;
		data.setClassIndex(dim);
		
		// get class sizes
		classSizes = new int[num_classes];
		for(int sample = 0; sample < num_data; sample++){
			int label = (int) data.get(sample).classValue();
			classSizes[label]++;
		}
		
		// initialize pairwise models & assign size of each based on the class sizes
		state.models = new ArrayList<PairwiseTrainingResult>();
		for(int i = 0; i < num_classes; i++){
			for(int j = (i+1); j < num_classes; j++){
				int size = classSizes[i] + classSizes[j];
				state.models.add(new PairwiseTrainingResult(params,size, new Tuple<Integer,Integer>(i,j)));
			}
		}
		
		// set the last label number as max label
		state.setLabelNumber(num_classes);
		
		// build the kernel evaluator
		eval = new KernelEvaluator(data, num_data, params.getGamma());
		
		// build the cache
		cache = new Cache(num_data,params,eval);
	}
	
	/**
	 * Train on a batch of instances.
	 */
	@Override
	public void trainOnInstances(Instances data) {
		intialize(data);
		// set up the environment for each pairwise model
		int totalSize = num_data;
		for(int i = 0; i < state.models.size(); i++){
			// reorder samples based on training pair
			Tuple<Integer,Integer> trainPair = state.models.get(i).trainingLabels;
			int size = reorderSamples(trainPair, totalSize);
			setCurrentSize(size);
			cache.setLabel(trainPair.second);
			cache.reset();
			
			// train binary OLLAWV model on current training pair
			cache.trainForCache();
			
			// update pairwise training model
			state.models.get(i).setAlphas(cache.getAlphas());
			state.models.get(i).setBias(cache.bias);
			state.models.get(i).setSvnumber(cache.svnumber-1);
			state.models.get(i).setSamples(cache.backwardOrder);
		}
		
		// sort the data so that all the support vectors are on top
		int freeOffset = 0;
		int[] mapping = cache.forwardOrder;
		for(Iterator<PairwiseTrainingResult> it = state.models.iterator(); it.hasNext();){
			PairwiseTrainingResult model = it.next();
			for(int j = 0; j < model.getSvnumber(); j++){
				int realOffset = mapping[model.getSample(j)];
				if(realOffset >= freeOffset){
					cache.swapSamples(realOffset, freeOffset);
					realOffset = freeOffset++;
				}
				model.setSample(j,realOffset);
			}
		}
		// set the SV number to be the total number of SVs per model with no repeats
		state.setSvNumber(freeOffset);
		
		// for debugging
		if(vOption.getValue() == 1){
			log.printBarrier();
			log.println(this.toString());
		}
	}
	
	/**
	 * Set current size of the pairwise model
	 */
	public void setCurrentSize(int size){
		cache.setCurrentSize(size);
	}
	
	/**
	 * Orders samples based on the current label training pair
	 * @param trainPair
	 * @param size
	 * @return number of samples for this training pair
	 */
	public int reorderSamples(Tuple<Integer,Integer> trainPair, int size){
		int first = trainPair.first;
		int second = trainPair.second;
		int train = 0;
		int test = size-1;		
		while(train <= test){
			while(train < size && (cache.eval.data.get(train).classValue() == first || cache.eval.data.get(train).classValue() == second)){
				train++;
			}
			while(test >= 0 && (cache.eval.data.get(test).classValue() != first && cache.eval.data.get(test).classValue() != second)){
				test--;
			}
			if(train < test){
				cache.swapSamples(train, test);
				
				train = train + 1;
				test = test - 1;
			}
		}
		return train;
	}
	
	/***
     * Calculates the class membership for the given test
     * instance (either 0 or 1). Result holds the negative class
     * probability in space result[0], and positive in result[1].
     *
     * @param	Instance 	unknown instance to be classified
     * @return 	double[] 	class probability, either 0 or 1
     */
	@Override
	public double[] getVotesForInstance(Instance inst) {
		double[] result = new double[num_classes];
		// initialize votes and evidence
		state.setVotes(new int[num_classes]);
		state.setEvidence(new double[num_classes]);
		
		if(cache != null){
			// calculate the kernel vector for all SVs
			int svnumber = state.getSvNumber();
			double[] G = new double[svnumber];
			eval.evalKernel(inst, svnumber, G);
			// for each model, calculate output and fill votes and evidence
			for(Iterator<PairwiseTrainingResult> it = state.models.iterator(); it.hasNext();){
				PairwiseTrainingResult model = it.next();
				double dec = getDecisionForModel(model,G);
				int label = dec > 0 ? model.trainingLabels.second : model.trainingLabels.first;
				state.votes[label]++;
				state.evidence[model.trainingLabels.first] += dec;
				state.evidence[model.trainingLabels.second] += dec;
			}
			// get the winning class (includes tied classes)
			int maxLabelId = 0;
			int maxVotes = 0;
			double maxEvidence = 0;
			for(int i = 0; i < state.getLabelNumber(); i++){
				if(state.votes[i] > maxVotes
					|| (state.votes[i] == maxVotes && state.evidence[i] > maxEvidence)){
					maxLabelId = i;
					maxVotes = state.votes[i];
					maxEvidence = state.evidence[i];
				}
			}
			//log.println(maxEvidence);
			result[maxLabelId]++;
			
//			if(maxLabelId == inst.classValue()){
//				// correct!
//			}
		}
		return result;
	}
	
	/**
	 * Get model output
	 * @param model
	 * @param G
	 * @return G*alpha + bias
	 */
	public double getDecisionForModel(PairwiseTrainingResult model, double[] G){
		double dec = model.getBias();
		for(int sample = 0; sample < model.getSvnumber(); sample++){
			dec += model.getAlpha(sample) * G[model.getSample(sample)];
		}
		return dec;
	}
	
	/**
	 * Returns a string of the model
	 * @return
	 */
	public String toString(){
		StringBuffer buff = new StringBuffer();
		buff.append("Total number of Pairwise Models: "+state.models.size()+"\n");
		buff.append("Max SV Number: "+state.getSvNumber()+"\n");
		
		for(int i = 0; i < state.models.size(); i++){
			PairwiseTrainingResult model = state.models.get(i);
			buff.append("\tPair "+i+":("+model.trainingLabels.first+","+model.trainingLabels.second+")\n");
			buff.append("\tNumber of Support Vectors: "+model.getSvnumber()+"\n");
			buff.append("\tBias: "+model.getBias()+"\n");
			buff.append("        Alphas: "+model.getAlphas().toString()+"\n");
			buff.append("        Samples: "+Arrays.toString(model.getSamples())+"\n");
			buff.append("----------------------------------\n");
		}
        return buff.toString();
	}

	@Override
	public void resetLearningImpl() {

	}

	@Override
	public void trainOnInstanceImpl(Instance inst) {
		log.print("ERROR: trainOnInstanceImpl was called and no implementation for this yet.\n");
		try {
			throw new Exception("trainOnInstanceImpl was called and no implementation for this yet.");
		} catch (Exception e) {
			e.printStackTrace();
		}
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
		buff.append("Dimensionality: "+this.dim+"\n");
		buff.append("Number of classes: "+this.num_classes+"\n");
		buff.append("Standardize data: "+this.standardize+"\n");
		buff.append("Seed: "+this.randomSeed+"\n");
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
	
	@Override
	public boolean isRandomizable() {
		return false;
	}

}
