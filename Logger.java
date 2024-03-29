package vcu.edu.datastreamlearning.ollawv;

import java.io.File;
import java.io.FileNotFoundException;
import java.io.FileWriter;
import java.io.IOException;
import java.io.PrintWriter;

import com.yahoo.labs.samoa.instances.Instances;

public class Logger extends PrintWriter {

	protected PrintWriter logger;
	protected int counter;
	/* Constructor to System.out output */
	Logger(){
		super(System.out,true);
		logger = new PrintWriter(System.out, true);
		counter = 0;
	}

	/* Constructor to file output */
	Logger(String filename) throws FileNotFoundException{
		super(new File(filename));
		File log = new File(filename);
		try {
			if(!log.exists()){
				log.createNewFile();
			}
			logger = new PrintWriter(new FileWriter(log));
		} catch (IOException e) {
			e.printStackTrace();
		}
	}

	/* Print barrier */
	public void printBarrier(){
		logger.printf("-----------------------------------------\n");
	}

	public void printInstancesToFile(Instances chunk){
		try {
			PrintWriter t = new PrintWriter("data"+counter+".txt");
			for(int i = 0; i < chunk.size(); i++){
				double label = chunk.get(i).classValue();
				t.print(label+" ");
				for(int j = 0; j < chunk.numAttributes()-1; j++){
					t.print(j+":"+chunk.get(i).value(j)+" ");
				}
				t.println();
			}
			counter++;
			t.close();


		} catch (FileNotFoundException e) {
			e.printStackTrace();
		}
	}

	/* Prints chunk of (Instances) data */
	public void printInstances(Instances chunk){
		for(int i = 0; i < chunk.size(); i++){
			double label = chunk.get(i).classValue();
			logger.print(label+" ");
			for(int j = 0; j < chunk.numAttributes()-1; j++){
				logger.print(j+":"+chunk.get(i).value(j)+" ");
			}
			logger.println();
		}
	}
}
