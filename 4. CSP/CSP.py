class CSP:
    def __init__(self, number_of_marks):
        """
        Here we initialize all the required variables for the CSP computation,
        according to the number of marks.
        """
        # Your code here
        self.number_of_marks = number_of_marks
        self.current_length = int(number_of_marks * (number_of_marks - 1) / 2)  # Update this line
        self.variables = [-1] * number_of_marks  # Update this line
        self.differences = [[-1] * number_of_marks] * number_of_marks  # Update this line

    def assign_value(self, i, v):
        """
        assign a value v to variable with index i
        """
        # Your code here
        self.variables[i] = v

    def check_constraints(self, actives) -> bool:
        """
        Here we loop over the differences array and update values.
        Meanwhile, we check for the validity of the constraints.
        """
        # Your code here
        active = self.number_of_marks
        diffs = set()
        for i in range(actives):
            if self.variables[i] == -1:
                continue
            for j in range(i + 1, actives):
                if self.variables[j] == -1:
                    continue
                if abs(self.variables[j] - self.variables[i]) in diffs:
                    return False
                diffs.add(abs(self.variables[j] - self.variables[i]))
        return True


    def backtrack(self, i, active = 0):
        """
         In this function we should loop over all the available values for
         the variable with index i, and recursively check for other variables values.
        """
        # Your code here
        start = self.variables[i - 1] + 1 if i > 0 else 0
        end = self.current_length
        if i == self.number_of_marks:
            if self.check_constraints(active):
                return True
        elif i == 0:
            start = 0
        for v in range(start, end):
            self.assign_value(i, v)
            if self.check_constraints(active + 1):
                if i == self.number_of_marks - 1:
                    return True
                else:
                    if not self.forward_check(i + 1):
                        #print("pruned")
                        continue
                    if self.backtrack(i + 1, active + 1):
                        return True
        self.variables[i] = -1
        return False


    def forward_check(self, i):
        """
        After assigning a value to variable i, we can make a forward check - if needed -
        to boost up the computing speed and prune our search tree.
        """
        dists = set()
        for k in range(self.number_of_marks):
            if self.variables[k] == -1:
                continue
            for j in range(k + 1, self.number_of_marks):
                if self.variables[j] == -1:
                    continue
                dists.add(abs(self.variables[j] - self.variables[k]))
        for j in range(i , self.number_of_marks):
            domain = set([x for x in range(self.variables[i - 1] + j - i ,self.current_length)])
            if len(dists) != 0 and len(domain - dists) == 0:
                return False
        return True

    def find_minimum_length(self) -> int:
        """
        This is the main function of the class.
        First, we start by assigning an upper bound value to variable current_length.
        Then, using backtrack and forward_check functions, we decrease this value until we find
        the minimum required length.
        """
        # Your code here
        self.current_length = int(self.number_of_marks * (self.number_of_marks - 1) / 2)
        while True:
            if self.backtrack(0):
                return self.current_length - 1
            self.current_length += 1
            self.variables = [-1] * self.number_of_marks

    def get_variables(self) -> list:
        """
        Get variables array.
        """
        # No need to change
        return self.variables


def main():
    # Your code here
    number_of_marks = 8
    csp = CSP(number_of_marks)
    print(csp.find_minimum_length())
    print(' '.join(map(str, csp.get_variables())))

if __name__ == '__main__':
    main()