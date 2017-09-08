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
		log.print("Number of chunks: %d\n", numChunks);
		log.print("Number of instances: %d\n", chunkSize);
		log.print("Dimensionality: %d\n", dim);
		log.print("Number of classes: %d\n", numClasses);
		log.print("Seed: %d\n", seed);
		
		/* Create stream & set stream options based on configuration. */
		RandomRBFGenerator stream = new RandomRBFGenerator();
		stream.instanceRandomSeedOption.setValue(seed);
		stream.modelRandomSeedOption.setValue(seed);
		stream.numClassesOption.setValue(numClasses);
		stream.numAttsOption.setValue(dim);
		stream.numCentroidsOption.setValue(2);
		stream.prepareForUse();
		
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
			if(chunkSize*numChunks < 25){
				log.printData(chunk, chunkSize, dim);
			}
			
			if(c != 0){
				// test
			}
			
			System.out.println("Chunk #"+(c+1)+":");
			//printData(chunk,10,dim);
			
		}
	}

}
