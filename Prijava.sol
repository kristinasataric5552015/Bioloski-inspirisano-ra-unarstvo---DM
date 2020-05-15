pragma solidity ^0.5.0;

contract Prijava {
  
    struct User {
    uint id;
    string ime;
    uint jmbg;
    string banka;
  }

  User[] users;
  uint nextId = 1; 
  
  function prijaviSe(string memory ime, uint jmbg, string memory banka) public {
    users.push(User(nextId, ime, jmbg,banka));
    nextId++;
    }
  
  function brojPrijavljenih() view public returns(uint) {
    return users.length;
  }
  
   function pronadji(uint jmbg) view public returns(string memory) {
    for (uint i = 0; i < users.length; i++){
      if(users[i].jmbg == jmbg) {
        return users[i].ime;
      }
    }
    revert('Korisnik sa naznacenim JMBG-om ne postoji!');
  }
 
}

