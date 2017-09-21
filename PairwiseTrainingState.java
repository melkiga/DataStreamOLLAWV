package vcu.edu.datastreamlearning.ollawv;

import java.util.List;

/**
 * Pairwise training state holds multiple pairwise OLLAWV models.
 * @author gabriellamelki
 *
 */
public class PairwiseTrainingState {
	protected List<PairwiseTrainingResult> models;
	private int svNumber;
	private int labelNumber;
	
	public PairwiseTrainingState(){
		models = null;
		svNumber = -1;
		labelNumber = -1;
	}
	
	public List<PairwiseTrainingResult> getModels() {
		return models;
	}
	public void setModels(List<PairwiseTrainingResult> models) {
		this.models = models;
	}
	public int getSvNumber() {
		return svNumber;
	}
	public void setSvNumber(int svNumber) {
		this.svNumber = svNumber;
	}
	public int getLabelNumber() {
		return labelNumber;
	}
	public void setLabelNumber(int labelNumber) {
		this.labelNumber = labelNumber;
	}
}
