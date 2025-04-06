import { Component, Input, input } from '@angular/core';
import { applyMixins } from 'rxjs/internal/util/applyMixins';
import { AppComponent } from '../app.component';

@Component({
  selector: 'app-user',
  imports: [],
  templateUrl: './user.component.html',
  styleUrl: './user.component.css'
})
export class UserComponent {
  name: string = 'Sebastian';
  session: boolean = true;

  getName() {
    this.name = prompt('Ingrese su Usuario') || 'Sebastian'; 
  }
}
