import { Component } from '@angular/core';
import { RouterOutlet } from '@angular/router';
import { UserComponent } from "./user/user.component";
import { AsignaturasComponent } from "./asignaturas/asignaturas.component";

@Component({
  selector: 'app-root',
  imports: [RouterOutlet, UserComponent, UserComponent, AsignaturasComponent, AsignaturasComponent],
  templateUrl: './app.component.html',
  styleUrl: './app.component.css'
})
export class AppComponent {
  title = 'sv_aprender';
  city = 'Duitama';
  
}
