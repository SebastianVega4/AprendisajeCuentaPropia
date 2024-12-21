import java.util.Scanner;

public class Main {
    public static void main(String[] args) {
        Scanner scanner = new Scanner(System.in);
        int interacciones = Integer.parseInt(scanner.nextLine());
        int contador = 0;
        StringBuilder salidadData = new StringBuilder();
        
        while (contador < interacciones) {
            int yearsTienen = Integer.parseInt(scanner.nextLine());
            String yearsWintSpaces = scanner.nextLine();
            String[] years = yearsWintSpaces.split(" ");
            
            int maxYear = Integer.MIN_VALUE;
            int minYear = Integer.MAX_VALUE;
            
            // Calcular el valor máximo y mínimo de los años
            for (String year : years) {
                int yearInt = Integer.parseInt(year);
                if (yearInt > maxYear) maxYear = yearInt;
                if (yearInt < minYear) minYear = yearInt;
            }
            
            int yearAlls = 1 + maxYear - minYear;
            salidadData.append(yearAlls - yearsTienen).append("\n");
            contador++;
        }
        
        System.out.print(salidadData.toString());
        scanner.close();
    }
}