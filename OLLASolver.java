/**
 * <!-- globalinfo-start --> 
 * Implements OLLA Worst Violator Solver, written by Gabriella Melki & VK.
 * <!-- globalinfo-end -->
 */

package vcu.edu.datastreamlearning.ollawv;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.Iterator;
import java.util.Random;

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
	 * Variables for returning probabilities (LOGLOSS) or winning class (HINGE)
	 */
    protected static final int HINGE = 0;
    protected static final int LOGLOSS = 1;
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
	public IntOption standardizeOption = new IntOption("standardize",'s',"Standardize data, 0 mean 1 stdv",1);
	/**
	 * To use hyper-parameter selection in first chunk
	 */
	public IntOption useHyperParameterOption = new IntOption("hyperparameter",'p',"Use hyper-parameter selection in first chunk.",1);
	/**
	 * Set seed for shuffling the data in hyper-parameter selection
	 */
	public IntOption seedOption = new IntOption("seed",'z',"Seed value for shuffling data.",1);
	/**
	 * Whether to randomize the data or not
	 */
	public IntOption randOption = new IntOption("rand",'r',"Option for randomizing the data.",1);
	/**
	 * Set the number of folds for CV during hyper-parameter selection
	 */
	public IntOption foldOption = new IntOption("fold",'f',"Option for setting CV folds for tuning hyper-parameters.",10);
	/**
	 * Set the number of folds for CV during hyper-parameter selection
	 */
	public IntOption lossOption = new IntOption("loss",'l',"Option for which loss function to use (hinge [0,1] or log [probabilities]).",0);
	
	
	/**
	 * Data Header Variables
	 */
	protected int[] classSizes;
	protected PairwiseTrainingState state;
	protected Standardize proc;
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
		// Set the state to be null
		this.state = new PairwiseTrainingState();
		// Set cache to be null
		cache = null;
		eval = null;
		proc = null;
	}

	/**
	 * Initializes the solver + the pairwise models
	 */
	public void initialize(Instances data){
		// set class index
		data.setClassIndex(data.numAttributes()-1);

		// get class sizes
		classSizes = new int[data.numClasses()];
		for(int sample = 0; sample < data.numInstances(); sample++){
			int label = (int) data.get(sample).classValue();
			classSizes[label]++;
		}

		// initialize pairwise models & assign size of each based on the class sizes
		state.models = new ArrayList<PairwiseTrainingResult>();
		for(int i = 0; i < data.numClasses(); i++){
			for(int j = (i+1); j < data.numClasses(); j++){
				int size = classSizes[i] + classSizes[j];
				state.models.add(new PairwiseTrainingResult(params, size, new Tuple<Integer,Integer>(i,j)));
			}
		}

		// set the last label number as max label
		state.setLabelNumber(data.numClasses());

		// build the kernel evaluator
		eval = new KernelEvaluator(data, data.numInstances(), params.getGamma());

		// build the cache
		cache = new Cache(data.numInstances(),params,eval);
	}

	/**
	 * Set up training environment for batch of instances
	 */
	@Override
	public void trainOnInstances(Instances data) {
		// Standardize data
		if(standardizeOption.getValue() == 1){
			proc = new Standardize();
			data = proc.convertInstances(data);
		}

		// Randomize & stratify the data
		if(randOption.getValue() == 1){
			Random rand = new Random(seedOption.getValue());
			data = new Instances(data);
			data.randomize(rand);
			data.stratify(foldOption.getValue());
		}

		// if this is the first chunk
		if(cache == null) {
			// print purpose and context if debug is enabled
			if(vOption.getValue() == 1){
				log.printBarrier();
				log.printf(getPurposeString());
				log.printf(getModelContextString());
				log.printBarrier();
			}
			// use hyper-parameter selection on first chunk		
			if(useHyperParameterOption.getValue() == 1){
				tuneHyperParameters(data);
			}
		}
		// initialize the cache and evaluator, then train
		initialize(data);
		pairwiseTraining();
	}

	/**
	 * Hyperparameter tuning for the first chunk of data. 
	 * No need to standardize the data here, because this only gets 
	 * called after initialize, wich standardizes the data.
	 * @param data
	 */
	public void tuneHyperParameters(Instances data){
		int folds = foldOption.getValue();
		double[] gamma = {0.001, 0.01, 0.1, 0.5, 1.0, 2.0, 4.0, 16.0};
		double[] tol = {0.01, 0.1, 1.0};
		// accuracy of fold k
		double accuracy = 0.0;
		// best accuracy
		double acc = 0.0;
		// indexes of best hyper-parameter for each fold
		int ii = 0; int jj = 0;
		// cross validation for parameter selection
		for(int i = 0; i < tol.length; i++){
			params.setTol(tol[i]);
			for(int j = 0; j < gamma.length; j++){
				params.setGamma(gamma[j]);
				for(int k = 0; k < folds; k++){
					// create train and test set
					Instances train = data.trainCV(folds, k);
					Instances test = data.testCV(folds, k);
					// initialize and train
					initialize(train);
					pairwiseTraining();
					// test model
					double[] result = new double[test.numClasses()];
					int correct = 0;
					for(int t = 0; t < test.numInstances(); t++){
						Instance test_inst = test.get(t);
						result = classify(test_inst);
						int true_output = (int) test_inst.classValue();
						if(result[true_output] == 1.0){
							correct++;
						}
					}
					// accumulated accuracy
					accuracy += correct;
				}
				// get mean fold accuracy
				accuracy = (double) 100*accuracy / (double) data.numInstances();
				// find indexes of best accuracy
				if(accuracy > acc){
					acc = accuracy;
					ii = i; jj = j;
				} else if(accuracy == acc && (ii >= i && jj <= j)){
					acc = accuracy;
					ii = i; jj = j;
				}
				// reset accuracy for next parameters
				accuracy = 0.0;
			}
		}

		if(vOption.getValue() == 1){
			log.printBarrier();
			log.println("Hyper-parameter tuning complete...");
			log.println("Best Tol: "+tol[ii]);
			log.println("Best Gamma: "+gamma[jj]);
		}

		// set winning parameters
		params.setGamma(gamma[jj]);
		params.setTol(tol[ii]);
	}

	/**
	 * Perform pairwise training
	 */
	public void pairwiseTraining(){
		// set up the environment for each pairwise model
		int totalSize = this.cache.problemSize;
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

	/**
	 * Calculates the class membership for the given test instance. 
	 * Result holds the negative class probability in space result[0], and positive in result[1].
	 * Returns 1 in the winning class' position in the result array.
	 * No need to standardize or randomize data because it has already been done in initialize.
	 * Tuning the hyper-parameters calls this only.
	 * @param data
	 * @return
	 */
	public double[] classify(Instance inst){
		double[] result = new double[inst.numClasses()];
		// initialize votes and evidence
		state.setVotes(new int[inst.numClasses()]);
		state.setEvidence(new double[inst.numClasses()]);

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
			result[maxLabelId]++;
		}
		return result;
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
		double[] result = new double[inst.numClasses()];
		// initialize votes and evidence
		state.setVotes(new int[inst.numClasses()]);
		state.setEvidence(new double[inst.numClasses()]);
		// standardize data if set
		if(standardizeOption.getValue() == 1){
			inst = proc.convertInstance(inst);
		}

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
			result[maxLabelId]++;
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
		buff.append("Data Standardization: "+standardizeOption.getValue()+"\n");
		buff.append("SVM Parameter C: "+params.getC()+"\n");
		buff.append("RBF Parameter gamma: "+params.getGamma()+"\n");
		buff.append("Using Bias: "+params.getBetta()+"\n");
		buff.append("Margin Tol: "+params.getTol()+"\n");
		buff.append("Number of Epochs: "+params.getEpochs()+"\n");
		buff.append("Using Change Detection: "+changeOption.getValue()+"\n");
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
