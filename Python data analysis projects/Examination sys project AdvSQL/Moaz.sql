--insertion to table question
go

create table Branch(
	id int primary key identity(1,1),
	[name] nvarchar(max) not null
);


create table Intake(
	id int primary key identity(1,1),
	[name] nvarchar(max) not null,
	branch_id int
);


create table Track(
	id int primary key identity(1,1),
	[name] nvarchar(max) not null,
	intake_id int
);

-----------------------------------------------------------------------------------------------

insert into question
values
	(1,'TF', 'Is the sky blue?', 'True'),
    (2, 'MCQ', 'What is the capital of France? A)Marseille  B)Lyon  C)Paris  D)Nice' , 'Paris'),
    (3, 'TF', 'Is the Earth flat?', 'False'),
	(4, 'MCQ', 'Which planet is known as the Red Planet?  A)Jupiter  B)Uranus  C)Mercury  D)Mars', 'Mars'),
	(5, 'MCQ', 'Which programming language is often used for web development?  A)JavaScript  B)C++  C)C#  D)Python', 'JavaScript'),
	(6, 'TXT', 'What is the largest mammal on Earth?', 'Blue Whale'),
	(7, 'MCQ', 'What is the capital of Egypt?  A)Alexandria  B)Cairo  C)Gizeh  D)Luxor', 'Cairo'),
	(8, 'TF', 'Sharks are mammals?','False'),
	(9, 'MCQ', 'What is the largest country in the world?  A)Russia  B)USA  C)Brazil  D)France', 'Russia'),
	(10, 'TF', 'South Africa has one capital?', 'False'),
	(11, 'MCQ', 'What is the tallest tower in the world?  A)Shanghai  B)Merdeka 118  C)Lotte World  D)Khalifa', 'Khalifa'),
	(12, 'TF', 'The longest river in the world is the river nile?', 'True'),
	(13, 'TXT', 'What is SQL refers to?', 'Structured query language'),
	(14, 'TXT', 'What is OPP refers to?', 'Object oriented programming'),
	(15, 'TF', 'Is the Earth the third planet from the sun?', 'True'),
	(16, 'MCQ', 'What is the highest mountain peak in the world?  A)K2  B)Everest  C)Lhotse  D)Makalu', 'Everest'),
	(17, 'MCQ', 'Which country has the largest uranium reserves in the world?  A)USA  B)South Africa  C)Australia  D)Nigiria','Australia'),
	(18, 'TXT', 'What is the largest ocean on Earth?', 'Pacific Ocean'),
	(19, 'MCQ', 'What is the capital of Japan?  A)Hiroshima  B)Tokyo  C)Nagasaki  D)kyoto', 'Tokyo'),
	(20, 'MCQ', 'What is the capital of Brazil?  A)sao paulo  B)Rio de Janeiro  C)Manaus  D)Brasília', 'Brasília'),
    (21, 'TF', 'Is C++ an extension of the C programming language?', 'True'),
    (22, 'MCQ', 'Which element does the chemical symbol "O" represent?  A)Oxygen  B)Nitrogen  C)Magnesium  D)Carbon', 'Oxygen'),
	(23, 'TXT', 'Define the algorithm?', 'Group of steps that lead to solution of the problem'),
	(24, 'TF', 'The Atlantic Ocean is the biggest ocean on Earth?', 'True'),
	(25, 'TF', 'China has the longest coastline in the world?', 'False'),
	(26, 'TF', 'Is a tomato a fruit?', 'True'),
	(27, 'MCQ', 'Which element has the chemical symbol "Fe"?  A)Nitrogen  B)Iron  C)Oxygen  D)Carbon', 'Iron'),
	(28, 'MCQ', 'What is the capital of Canada?  A)Toronto  B)Ottawa  C)Victoria  D)Vancouver', 'Ottawa'),
    (29, 'TF', 'Is a panda a herbivore?', 'True'),
	(30, 'TXT', 'What is the largest desert in the world?', 'Antarctica (Cold Desert)'),
	(31, 'TXT', 'What is the capital of South Korea?', 'Seoul'),
	(32, 'TXT', 'What is the fastest animal in the world?', 'The cheetah'),
	(33, 'TF', 'Human skin regenerates every week?', 'False'),
	(34, 'MCQ', 'How many Abbasid caliphs were there?  A)37  B) 34  C)35  D)39', '37'),
	(35, 'TF', 'Brazil is the only nation to have played in every World Cup finals tournament?', 'True'),
	(36, 'TXT', 'Where is the Umayyad Mosque located?', 'Damascus'),
	(37, 'MCQ', 'How many valves are in the human heart?  A)3  B)4  C)5  D)6', '4'),
	(38, 'TF', 'An octopus has seven hearts?', 'False'),
	(39, 'TXT', 'Who is the first woman to win a Nobel?', 'Marie Curie'),
	(40, 'MCQ', 'What is the most abundant element in the human body?  A)Carbon  B)Oxygen  C)Nitrogen  D)Magnesium' ,'Oxygen'),
	(41, 'MCQ', 'What is the capital of England?  A)Liverpool  B)Birmingham  C)London  D)Manchester', 'London');


	create procedure Add_Student_To_Course_PROC
    @StudentID int, @CourseID int
as
begin
    if not exists (select 1 from [dbo].[Student] where id = @StudentID)
    begin
        print('Student does not exist.')
        return
    end

    if not exists (select 1 from [dbo].[Course] where c_id = @CourseID)
    begin
        print('Course does not exist.')
        return
    end

    if exists (select 1 from [dbo].[studentCourse] where StudentID = @StudentID and CourseID = @CourseID)
    begin
        print('Student is already enrolled in this course.')
        return
    end

    insert into studentCourse (StudentID, CourseID)
    values (@StudentID, @CourseID);

    print('Student added to course successfully.');
end;

---------(test)

exec Add_Student_To_Course_PROC 2, 4;
exec Add_Student_To_Course_PROC 3, 5;
exec Add_Student_To_Course_PROC 1, 8;
exec Add_Student_To_Course_PROC 4, 3;
exec Add_Student_To_Course_PROC 1, 7;
exec Add_Student_To_Course_PROC 4, 7;


------------------------------------------------------------------------------------------

-- index on table branch
create nonclustered index Banch_name_index
on Branch(name)


-- index on table intake
create nonclustered index Intake_name_index
on Intake(name)


--index on table tack
create nonclustered index Track_name_index
on Track(name)


--index on table student
create nonclustered index Student_name_index
on Student(name)

create nonclustered index Student_univetsity_index
on Student(university)

create nonclustered index Student_Phone_index
on Student(Phone)


--index on table instructor
create nonclustered index Instructor_name_index
on Instructor(name)


--index on table Course
create nonclustered index Course_name_index
on Course(name)

create nonclustered index Course_minDegree_index
on Course(minDegree)

create nonclustered index Course_maxDegree_index
on Course(maxDegree)


--index on table exam
create nonclustered index Exam_type_index
on Exam(type)

create nonclustered index Exam_year_index
on Exam(year)


--index on table question
create nonclustered index Question_type_index
on Question(type)

create nonclustered index Question_questionText_index
on Question(questionText)

create nonclustered index Question_questionAnswer_index
on Question(questionAnswer)