package vcu.edu.datastreamlearning.ollawv;

import com.yahoo.labs.samoa.instances.Instances;

import moa.streams.generators.RandomRBFGenerator;

public class OLLAWVTester {
	
	public static void main(String[] args){
		Logger log = new Logger();
		/* Set up data configuration. */
		int numChunks = 2;
		int chunkSize = 20;
		int dim = 2;
		int seed = 1;
		int numClasses = 2;
		
		/* Print configuration. */
		log.printFormatted("Number of chunks: %d\n", numChunks);
		log.printFormatted("Number of instances: %d\n", chunkSize);
		log.printFormatted("Dimensionality: %d\n", dim);
		log.printFormatted("Number of classes: %d\n", numClasses);
		log.printFormatted("Seed: %d\n", seed);
		log.printBarrier();
		
		/* Create stream & set stream options based on configuration. */
		RandomRBFGenerator stream = new RandomRBFGenerator();
		stream.instanceRandomSeedOption.setValue(seed);
		stream.modelRandomSeedOption.setValue(seed);
		stream.numClassesOption.setValue(numClasses);
		stream.numAttsOption.setValue(dim);
		stream.numCentroidsOption.setValue(2);
		stream.instanceRandomSeedOption.setValue(seed);
		stream.numAttsOption.setValue(dim);
		stream.prepareForUse();
		
		/* Initialize model */
		OLLASolver model = new OLLASolver();
		model.cOption.setValue(1.0);
		model.gOption.setValue(0.5);
		model.changeOption.setValue(0);
		model.vOption.setValue(1);
		model.setModelContext(stream.getHeader());
		
		/* Prequential Test-then-train on each chunk */
		int numSamplesProcessed = 0;
		for(int c = 0; c < numChunks; c++){
			/* Build each chunk */
			Instances chunk = new Instances(stream.getHeader(), chunkSize);
			for(int i = numSamplesProcessed; i < (chunkSize + numSamplesProcessed); i++){
				chunk.add(stream.nextInstance().getData());
			}
			numSamplesProcessed += chunkSize;
			
			/* Print data */
			if(chunkSize < 25){
				log.printBarrier();
				log.printFormatted("CHUNK #%d\n",c);
				log.printData(chunk, chunkSize, dim);
				log.printBarrier();
			}
			
			if(c != 0){
				// test
				// TODO: write up test code
				// TODO: write up timer for testing
			}
			
			// train
			// TODO: write up timer for training
			
		}
	}

}
