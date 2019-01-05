bool doubleL(string theString){
    theString.erase(remove(theString.begin(), theString.end(), ' '), theString.end());
    if(theString.find("ll")){
        return true;
    }
    else{
        return false;
    }
}