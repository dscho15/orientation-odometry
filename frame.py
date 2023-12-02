class Frame:
    
    def __init__(self, id) -> None:
        
        self.id = id
    
    def __hash__(self) -> int:
        return self.id
    
    