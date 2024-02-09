create database Examination;
----------------------------
go 
use [Examination]
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
	intake_id int-
);


create table Student(
	id int primary key identity(1,1),
	[name] nvarchar(max) not null,
	univeristy nvarchar(max) not null,
	Phone char(11) not null,
	gradDate date not null,
	track_id int 
);


create table Instructor(
	ins_id int primary key identity(1,1),
	[name] nvarchar(max) not null
);


create table Course(
	c_id int primary key identity(1,1),
	[name] nvarchar(max) not null,
	[desc] nvarchar(max) not null,
	minDegree int not null,
	maxDegree int not null,
	instructorID int 
);


create table studentCourse(
	studentID int,
	courseID int
	primary key(studentID,courseID)
);


create table Exam(
	e_id int primary key identity(1,1),
	[type] bit not null,       --exam or corrective
	[year] date not null,
	allowance nvarchar(max) not null,
	startTime datetime not null,
	endTime datetime not null,
	course_id int, 
	instructor_id int
);


create table question
(
	ID int primary key,
	[type] nchar(3), -- TF, MCQ, TXT
	questionText nvarchar(max),
	questionAnswer nvarchar(max) not null
);


create table questionExam
(
	questionID int,
	examID int,
	degree int not null,
	primary key(questionID,examID)
);


create table examStudent
(
	examID int,
	studentID int,
	question_id int,
	primary key(examID,studentID),
	answer nvarchar(max),
	score int
);


----------------
--Relations


alter table Intake 
add constraint intake_branch
foreign key (branch_id)
references Branch(id);


alter table Track 
add constraint track_intake
foreign key (intake_id)
references Intake(id);


alter table Student 
add constraint student_track
foreign key (track_id)
references Track(id);


alter table Course 
add constraint instructor_course
foreign key (instructorID)
references Instructor(ins_id);


alter table studentCourse 
add constraint studentCourse_student
foreign key (studentID)
references Student(id);


alter table studentCourse 
add constraint studentCourse_course
foreign key (courseID)
references Course(c_id);


alter table Exam 
add constraint exam_course
foreign key (course_id)
references Course(c_id);


alter table Exam 
add constraint exam_instructor
foreign key (instructor_id)
references Instructor(ins_id);


alter table questionExam 
add constraint questionExam_question
foreign key (questionID)
references question(ID);


alter table questionExam 
add constraint questionExam_exam
foreign key (examID)
references Exam(e_id);


alter table examStudent 
add constraint examStudent_exam
foreign key (examID)
references Exam(e_id);


alter table examStudent 
add constraint examStudent_student
foreign key (examID)
references Student(id);


alter table examStudent 
add constraint examStudent_question
foreign key (question_id)
references question(ID);


---------------------------------

--insertion to table question
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

go


/*
-- Training manager can add and edit:
Branches, 
tracks 
intake.
*/

-- display proc
create proc disp_branch_proc
as
begin
	select * from [dbo].[Branch];
end

go
create proc disp_track_proc
as
begin
	select * from [dbo].[Track];
end

go
create proc disp_intake_proc
as
begin
	select * from [dbo].[Intake];
end


-- add proc

go
create proc add_branch_proc
(@name nvarchar(max))
as
begin
	insert into [dbo].[Branch]
	values (@name);
end

go
create proc add_intake_proc
(@name nvarchar(max), @branch_name nvarchar(max))
as
begin
	insert into [dbo].[Intake]
	values (@name, (select id from Branch where name = @branch_name));
end



-- edit proc

go
create proc edit_branch_name_proc
(@oldname nvarchar(max), @newname nvarchar(max))
as
begin
	update [dbo].[Branch]
	set name = @newname
	where name = @oldname;
end

go
create proc edit_intake_name_proc 
(@oldname nvarchar(max), @newname varchar(max), @branch_name nvarchar(max))
as 
begin
	update 	[dbo].[Intake]
	set name = @newname
	where name = @oldname and branch_id = (select id from Branch where name = @branch_name)
end

go
create proc edit_track_name_proc
(@oldname nvarchar(max), @newname nvarchar(max), @intake_name nvarchar(max), @branch_name nvarchar(max))
as
begin
	update [dbo].[Track]
	set name = @newname
	where name = @oldname and intake_id = (select id from Intake where name = @intake_name and
	branch_id = (select id from Branch where name = @branch_name))
end

-- add specific track in one branch
go
create proc add_track_branch_proc
(@name nvarchar(max), @intake_name nvarchar(max), @branch_name nvarchar(max))
as 
begin
	insert into [dbo].[Track]
	values(
	@name, 
	(select i.id from Intake i join Branch b on b.id = i.branch_id 
	where b.name = @branch_name and i.name = @intake_name)
	)
end


-- display all
go
create proc disp_all_proc
as
begin
	select b.id 'branch id',b.name 'branch name', 
	i.id 'intake id', i.name 'intake name',
	t.intake_id 'track id', t.name 'track name'
	from Branch b
	join intake i
	on i.branch_id = b.id
	join track t
	on t.intake_id = i.id  
end

go

/*
Training manager can 
add students, and
define their personal data, 
intake, branch, and track.
*/

-- disp students
create proc disp_student_proc
as
begin
	select * from [dbo].[Student];
end


-- add student
go
create proc add_student_proc
(
@name nvarchar(max), @uni nvarchar(max), @phone char(11),
@gradDate date, 
@track_name nvarchar(max), @intake_name nvarchar(max), @branch_name nvarchar(max)
)
as
begin
	insert into [dbo].[Student]
	values (@name, @uni, @phone, @gradDate, 
	(
	select t.id
	from Track t
	join Intake i
	on i.id = t.intake_id
	join Branch b
	on b.id = i.branch_id
	where t.name = @track_name and i.name = @intake_name and b.name = @branch_name
	));
end


-- delete student
go
create proc delete_student_proc
(@name nvarchar(max), 
@track_name nvarchar(max), @intake_name nvarchar(max), @branch_name nvarchar(max))
as 
begin
	delete from [dbo].[Student]
	where name = @name and track_id = 
	(
	select t.id
	from Track t
	join Intake i
	on i.id = t.intake_id
	join Branch b
	on b.id = i.branch_id
	where t.name = @track_name and i.name = @intake_name and b.name = @branch_name
	);
end


exec dbo.add_branch_proc 'Smart Village'
exec dbo.add_branch_proc 'Assuit'
exec dbo.add_branch_proc 'Ismailia'
exec dbo.add_branch_proc 'Menofia'
exec dbo.add_branch_proc 'Alexandria'
exec dbo.add_branch_proc 'Aswan'
exec dbo.add_branch_proc 'Mansoura'
exec dbo.add_branch_proc 'Minya'

exec dbo.add_intake_proc 'one', 'Smart Village' -- adds intakes in branches
exec dbo.add_intake_proc 'two', 'Smart Village'
exec dbo.add_intake_proc 'three', 'Smart Village'
exec dbo.add_intake_proc 'four', 'Smart Village'
exec dbo.add_intake_proc 'five', 'Smart Village'

exec dbo.add_intake_proc 'one', 'Smart Village' -- adds intakes in branches
exec dbo.add_intake_proc 'two', 'Smart Village'
exec dbo.add_intake_proc 'three', 'Smart Village'
exec dbo.add_intake_proc 'four', 'Smart Village'
exec dbo.add_intake_proc 'five', 'Smart Village'

exec dbo.add_intake_proc '2020Q1', 'Assuit' -- only smart village has the 9 months here
exec dbo.add_intake_proc '2020Q2', 'Assuit'
exec dbo.add_intake_proc '2021Q1', 'Assuit'
exec dbo.add_intake_proc '2021Q2', 'Assuit'
exec dbo.add_intake_proc '2022Q1', 'Assuit'
exec dbo.add_intake_proc '2022Q2', 'Assuit'
exec dbo.add_intake_proc '2022Q2', 'Assuit'
exec dbo.add_intake_proc '2023Q1', 'Assuit'
exec dbo.add_intake_proc '2023Q2', 'Assuit'

exec dbo.add_intake_proc '2020Q1', 'Ismailia'
exec dbo.add_intake_proc '2020Q2', 'Ismailia'
exec dbo.add_intake_proc '2021Q1', 'Ismailia'
exec dbo.add_intake_proc '2021Q2', 'Ismailia'
exec dbo.add_intake_proc '2022Q1', 'Ismailia'
exec dbo.add_intake_proc '2022Q2', 'Ismailia'
exec dbo.add_intake_proc '2022Q2', 'Ismailia'
exec dbo.add_intake_proc '2023Q1', 'Ismailia'
exec dbo.add_intake_proc '2023Q2', 'Ismailia'

exec dbo.add_intake_proc '2020Q1', 'Menofia'
exec dbo.add_intake_proc '2020Q2', 'Menofia'
exec dbo.add_intake_proc '2021Q1', 'Menofia'
exec dbo.add_intake_proc '2021Q2', 'Menofia'
exec dbo.add_intake_proc '2022Q1', 'Menofia'
exec dbo.add_intake_proc '2022Q2', 'Menofia'
exec dbo.add_intake_proc '2022Q2', 'Menofia'
exec dbo.add_intake_proc '2023Q1', 'Menofia'
exec dbo.add_intake_proc '2023Q2', 'Menofia'

exec dbo.add_intake_proc '2020Q1', 'Alexandria'
exec dbo.add_intake_proc '2020Q2', 'Alexandria'
exec dbo.add_intake_proc '2021Q1', 'Alexandria'
exec dbo.add_intake_proc '2021Q2', 'Alexandria'
exec dbo.add_intake_proc '2022Q1', 'Alexandria'
exec dbo.add_intake_proc '2022Q2', 'Alexandria'
exec dbo.add_intake_proc '2022Q2', 'Alexandria'
exec dbo.add_intake_proc '2023Q1', 'Alexandria'
exec dbo.add_intake_proc '2023Q2', 'Alexandria'

exec dbo.add_intake_proc '2020Q1', 'Aswan'
exec dbo.add_intake_proc '2020Q2', 'Aswan'
exec dbo.add_intake_proc '2021Q1', 'Aswan'
exec dbo.add_intake_proc '2021Q2', 'Aswan'
exec dbo.add_intake_proc '2022Q1', 'Aswan'
exec dbo.add_intake_proc '2022Q2', 'Aswan'
exec dbo.add_intake_proc '2022Q2', 'Aswan'
exec dbo.add_intake_proc '2023Q1', 'Aswan'
exec dbo.add_intake_proc '2023Q2', 'Aswan'

exec dbo.add_intake_proc '2020Q1', 'Mansoura'
exec dbo.add_intake_proc '2020Q2', 'Mansoura'
exec dbo.add_intake_proc '2021Q1', 'Mansoura'
exec dbo.add_intake_proc '2021Q2', 'Mansoura'
exec dbo.add_intake_proc '2022Q1', 'Mansoura'
exec dbo.add_intake_proc '2022Q2', 'Mansoura'
exec dbo.add_intake_proc '2022Q2', 'Mansoura'
exec dbo.add_intake_proc '2023Q1', 'Mansoura'
exec dbo.add_intake_proc '2023Q2', 'Mansoura'

exec dbo.add_intake_proc '2020Q1', 'Minya'
exec dbo.add_intake_proc '2020Q2', 'Minya'
exec dbo.add_intake_proc '2021Q1', 'Minya'
exec dbo.add_intake_proc '2021Q2', 'Minya'
exec dbo.add_intake_proc '2022Q1', 'Minya'
exec dbo.add_intake_proc '2022Q2', 'Minya'
exec dbo.add_intake_proc '2022Q2', 'Minya'
exec dbo.add_intake_proc '2023Q1', 'Minya'
exec dbo.add_intake_proc '2023Q2', 'Minya'

exec add_track_branch_proc 'OS','one', 'Smart Village' -- add tracks in intake and branches
exec add_track_branch_proc 'SD','two', 'Smart Village'
exec add_track_branch_proc 'DS','three', 'Smart Village'
exec add_track_branch_proc 'DS','four', 'Smart Village'
exec add_track_branch_proc 'DA','five', 'Smart Village'

exec add_track_branch_proc '.net','2020Q1', 'Assuit'
exec add_track_branch_proc 'python','2020Q2', 'Assuit'

exec add_track_branch_proc '.net','2021Q1', 'Ismailia'
exec add_track_branch_proc 'python','2021Q2', 'Ismailia'

exec add_track_branch_proc '.net','2022Q1', 'Menofia'
exec add_track_branch_proc 'python','2022Q2', 'Menofia'

exec add_track_branch_proc '.net','2022Q1', 'Alexandria'

exec add_track_branch_proc '.net','2022Q1', 'Aswan'

exec add_track_branch_proc '.net','2022Q1', 'Mansoura'


exec add_track_branch_proc '.net','2022Q1', 'Minya'
exec add_track_branch_proc 'BI','2022Q2', 'Minya'

exec add_track_branch_proc 'BI','2023Q1', 'Minya'
exec add_track_branch_proc 'BI','2023Q2', 'Minya'

exec add_track_branch_proc 'python', '2023Q2', 'Minya'
exec add_track_branch_proc '.net', '2023Q2', 'Minya'
exec add_track_branch_proc 'web', '2023Q2', 'Minya'

exec add_student_proc 'Rana', 'Minia Uni', '01097363250', '2023', 'BI', '2023Q2', 'Minya'
exec add_student_proc 'Mohamed', 'Assuit Uni', '01012345678', '2020', 'python', '2023Q2', 'Minya'
exec add_student_proc 'Ahmed', 'Cairo Uni', '01112345678', '2021', '.net', '2023Q2', 'Minya'
exec add_student_proc 'Moaz', 'Alexandria Uni', '01212345678', '2022', 'web', '2023Q2', 'Minya'

go

create proc add_instructor_proc @name nvarchar(max)
as
begin
	insert into [dbo].[Instructor]
	values (@name);
	PRINT 'Instructor Inserted: ' + @name + ' is inserted.';
end;

go

create proc delete_instructor_proc 
@ins_id int,
@name nvarchar(max)
as
begin
    IF EXISTS (SELECT ins_id FROM [Instructor] WHERE ins_id = @ins_id AND name = @name)
    BEGIN
        DELETE FROM [Instructor] WHERE ins_id = @ins_id AND name = @name;
        PRINT 'Instructor with ins_id ' + CAST(@ins_id AS NVARCHAR(10)) + ' is deleted';

    END
    ELSE
        PRINT 'Instructor with ins_id ' + CAST(@ins_id AS NVARCHAR(10)) + ' does not exist';
end;

go

create proc edit_instructor_proc 
@old_name nvarchar(max), @new_name nvarchar(max), @id int
as
begin
    if exists (select @id from [Instructor] where ins_id= @id and [name] = @old_name)
    begin
        update [Instructor]
        set [Name] = @new_name
           where ins_id = @id and [name] = @old_name;

        print 'Instructor: ' + @new_name + ' is updated';
    end
    else
        print 'Instructor does not exist';
end;

go

create proc disp_instructor_proc
as
begin
	select * from [dbo].[Instructor];
end;


exec [dbo].[add_instructor_proc] 'ahmed'
exec [dbo].[add_instructor_proc] 'ali'
exec [dbo].[add_instructor_proc] 'khaled'
exec [dbo].[add_instructor_proc] 'abdo'
exec [dbo].[add_instructor_proc] 'saeed'
exec [dbo].[add_instructor_proc] 'mostafa'
exec [dbo].[add_instructor_proc] 'moaaz'
exec [dbo].[add_instructor_proc] 'taha'

exec [dbo].[edit_instructor_proc] 'taha' , 'rami', 8
exec [dbo].[disp_instructor_proc] -- displays the edit

exec [dbo].[delete_instructor_proc] 16, 'taha' -- delete instructor


go
create proc add_course_proc
    @course_name nvarchar(max),
    @description nvarchar(max),
    @min_degree int,
	@max_degree int,
	@instructor_id int
as
begin
    insert into Course
    values (@course_name, @description, @min_degree, @max_degree, @instructor_id);
end;


go
create proc delete_course_proc @course_id int
as
begin
    delete from Course
    where c_id = @course_id;
end;


go
create proc edit_course_proc
    @course_name nvarchar(max),
    @description nvarchar(max),
    @min_degree int,
	@max_degree int,
	@instructor_id int
as
begin
    update Course
    set [name] = @course_name,
        [desc] = @description,
		minDegree = @min_degree,
        maxDegree = @max_degree,
		instructorID = @instructor_id
    where name = @course_name;
end;



go
create function get_course_info_by_name_fun(@course_name nvarchar(max))
returns table
as
return (
    select [name], [desc], minDegree, maxDegree
    from Course
    where [name] = @course_name
);

go
create or alter function get_instructor_courses_name_id_func(@ins_id int, @name nvarchar(max))
returns table
as
return
(
	select ins.[name] 'instructor name' , co.[name] 'course name', [desc], minDegree, maxDegree
	from Course co
	join Instructor ins
	on ins.ins_id = co.instructorID
	where ins.name = @name
);

--test:
select * from [dbo].[get_course_info_by_name_fun]('SQL');
select * from [dbo].[get_instructor_courses_name_id_func](1, 'ahmed')



go
create proc disp_course_proc
as
begin
	select * from [dbo].[Course];
end;

exec add_course_proc 'Excel', 'Basic excel concepts', 0, 100, 24;
exec add_course_proc 'C#', 'Basic C# concepts', 0, 100, 23;
exec add_course_proc 'SQL', 'Basic database concepts', 0, 100, 8;
exec add_course_proc 'Python', 'Fundamentals of programming', 0, 100, 1;
exec add_course_proc 'Power BI', 'Building interactive dashboards', 0, 100, 2;
exec add_course_proc 'Data Science Fundamentals', 'Introduction to data analysis', 0, 100, 3;
exec add_course_proc 'Network Security', 'Securing computer networks', 0, 100, 8;
exec add_course_proc 'Data Mining', 'Overview of statistics and AI techniques', 0, 100, 1;
exec add_course_proc 'Web design basics', 'Overview of HTML and CSS', 0, 100,4;


exec delete_course_proc 5; -- delete course with id 5

exec edit_course_proc 'SQL' , 'fundamentals of database concepts', 0, 100, 8


--- this is all for 2023Q2 ONLY
go
create or alter view courses_for_BI
as
select
c_id, c.name, t.id
from Course c, Track t 
where c_id in (1,2,3,4,6) and t.id = 43;

go
select * from courses_for_BI

go
create or alter view courses_for_python
as 
select 
c_id, c.name, t.id
from Course c, Track t 
where c_id in (1,2,4) and t.id = 45;

go
create or alter view courses_for_dot_net
as 
select 
c_id, c.name, t.id
from Course c, Track t 
where c_id in (1,7) and t.id = 46;


go
create or alter view courses_for_web
as 
select 
c_id, t.id
from Course c, Track t  
where c_id in (7) and t.id = 47;



go
create or alter proc insert_course_student_BI_proc
as
begin
	insert into studentCourse (studentID, courseID)
		select s.id, c_id
		from 
		Student s
		join courses_for_BI cBI
		on cBI.id = s.track_id
	
end

go
create or alter proc insert_course_student_py_proc
as
begin
	insert into studentCourse (studentID, courseID)
		select s.id, c_id
		from 
		Student s
		join courses_for_python cPY
		on cPY.id = s.track_id
end

go
create or alter proc insert_course_student_dot_proc
as
begin
	insert into studentCourse (studentID, courseID)
		select s.id, c_id
		from 
		Student s
		join courses_for_dot_net cDOT
		on cDOT.id = s.track_id
end

go
create or alter proc insert_course_student_web_proc
as
begin
	insert into studentCourse (studentID, courseID)
		select s.id, c_id
		from 
		Student s
		join courses_for_web cWEB
		on cWEB.id = s.track_id
end


go 
create or alter proc insert_all_student
as
begin
	exec insert_course_student_BI_proc
	exec insert_course_student_py_proc
	exec insert_course_student_dot_proc
	exec insert_course_student_web_proc
end

exec insert_all_student

go 
create function view_student_courses_func ()
returns table
as
return 
(
	select s.name 'Student name', c.name 'Course name'
	from
	Student s
	join studentCourse sc
	on sc.studentID = s.id
	join Course c
	on sc.courseID = c.c_id
)
go

--- proc for making exam before assigning questions

create proc make_exam_proc
(@ins_id int, @c_id int, @type bit,
@year date, @allownace nvarchar(max),
@startTime datetime, @endTime datetime)
as
begin
	if @ins_id = (select instructorID from Course where c_id = @c_id)
	begin
		insert into Exam
		values (@type, @year, @allownace, @startTime, @endTime, @c_id, @ins_id)

		print 'to chose questions by selecting number of questions of each type and ' 
		+ 'the system selects the questions random '
		+ 'exec chose_Q_random_proc'
		print 'to manually chose and view questions exec chose_Q_manual_proc'

	end
	else
		print 'Can only make exam for your course'
end




go
create or alter proc chose_Q_random_proc
(@exam_id int,@c_id int,
@NMCQ int, @NTF int, @NTXT int, 
@MCQ_degree int, @TF_degree int, @TXT_degree int)
as
begin
		if @exam_id in (select e_id from Exam)
		begin
			declare @MCQ_count as int = 
			(select count(id)
			from question
			group by type
			having type = 'MCQ')

			declare @TF_count as int = 
			(select count(id)
			from question
			group by type
			having type = 'TF')

			declare @TXT_count as int = 
			(select count(id)
			from question
			group by type
			having type = 'TXT')

			if (@MCQ_count >= @NMCQ and @TF_count >= @NTF and @TXT_count >= @NTXT)
			begin
				if ((@NMCQ * @MCQ_degree) + (@NTF * @TF_degree) + (@NTXT * @TXT_degree))
				< (select maxDegree from Course where c_id = @c_id)
				begin
					if @NMCQ > 0
					begin
						insert into questionExam
							select top(@NMCQ) id, @exam_id, @MCQ_degree
							from question
							where type = 'MCQ'
							order by NEWID()
					end
					if @NTF > 0
					begin
						insert into questionExam
							select top(@NTF) id, @exam_id, @TF_degree
							from question
							where type = 'TF'
							order by NEWID()
					end
					if @NTXT > 0
					begin
						insert into questionExam
							select top(@NMCQ) id, @exam_id, @TXT_degree
							from question
							where type = 'TXT'
							order by NEWID()
					end
					select * from questionExam
				end
				else
					print 'sum of degree must be lower than max degree in course'
			end
			else 
				print 'chose lower number of questions'
		end
		else
		print 'make exam first'
end

go
create or alter proc disp_questions_proc

as 
begin
	select * from question
end


go 
create or alter proc make_exam_manual_proc(
@exam_id int, @c_id int,
@Question_id_string nvarchar(max), 
@question_degree_string nvarchar(max))
as
begin
	if @exam_id in (select e_id from Exam)
	begin
		declare @value_question table (id int) 
		insert into @value_question select cast(value as int) from string_split(@Question_id_string, ',')
		if (select max(id) from @value_question) < (select max(ID) from question)
		begin
			declare @maxdegree int = (select maxDegree from Course where c_id = @c_id)
			declare @value_degree table (degree int)
			insert into @value_degree select cast(value as int) from string_split(@question_degree_string, ',')
			declare @sumdegree int = (select sum(degree) from @value_degree)
			if @sumdegree < @maxdegree
			begin
				insert into questionExam
					select
					 i.id, @exam_id, d.degree
					from
					(select id,ROW_NUMBER() OVER (ORDER BY id) AS rn from @value_question) as i
					full outer join 
					(select degree, ROW_NUMBER() OVER (ORDER BY degree) AS rn from @value_degree) as d
					on i.rn = d.rn
			end
			else 
				print 'cant make max degrees bigger than max degree for course'
		end
		else
		print 'chose ids from question table'
	end
	else
	print 'make exam first'
end


--test
exec make_exam_proc @ins_id = 8, @c_id = 1, @type = 0,
@year = '2023', @allownace = 'half hour extra', 
@startTime = '2023-12-1 12:00', @endTime = '2023-12-1 14:00'


declare @eid int = (select IDENT_CURRENT('Exam'))
-- test make exam 
exec chose_Q_random_proc 
@exam_id = @eid, @c_id = 1, 
@NMCQ = 4, @NTF = 4, @NTXT = 4,
@MCQ_degree = 1, @TF_degree = 2, @TXT_degree = 3

go
exec make_exam_proc @ins_id = 1, @c_id = 2, @type = 0,
@year = '2023', @allownace = 'half hour extra', 
@startTime = '2023-12-2 12:00', @endTime = '2023-12-2 14:00'

declare @eid int = (select IDENT_CURRENT('Exam'))
exec make_exam_manual_proc @exam_id = @eid, @c_id = 2,
@Question_id_string = '1,2,3,4,5,6,7', @question_degree_string = '1,2,3,4,5,6,7'

go
create or alter proc select_students_for_exam_proc @student_id int, @exam_id int
as
begin
    if not exists (select 1 from Student where id = @student_id)
    begin
        print 'Student does not exist';
        return;
    end

    if not exists (select 1 from Exam where e_id = @exam_id)
    begin
        print 'Exam does not exist';
        return;
    end

    if exists (select 1 from examStudent where studentID = @student_id and  examID= @exam_id)
    begin
        print 'Student already assigned to this exam';
        return;
    end

    -- add student to the exam
    insert into examStudent (studentID, examID) values (@student_id, @exam_id);
    print 'Student added to exam successfully';
end;

--test:
exec select_students_for_exam_proc 1, 1
exec select_students_for_exam_proc 2, 1





--Instructor can select students that can do specific exam
create or alter proc select_students_for_exam_proc @student_id int, @exam_id int
as
begin
    if not exists (select 1 from Student where id = @student_id)
    begin
        print 'Student does not exist';
        return;
    end

    if not exists (select 1 from Exam where e_id = @exam_id)
    begin
        print 'Exam does not exist';
        return;
    end

    if exists (select 1 from examStudent where studentID = @student_id and  examID= @exam_id)
    begin
        print 'Student already assigned to this exam';
        return;
    end

    -- add student to the exam
    insert into examStudent (studentID, examID) values (@student_id, @exam_id);
    print 'Student added to exam successfully';
end;


--test and add data 
exec select_students_for_exam_proc 1, 3
exec select_students_for_exam_proc 4, 3
exec select_students_for_exam_proc 2, 3
exec select_students_for_exam_proc 6, 3
exec select_students_for_exam_proc 2, 3





--checks if student has this exam and if this the time for it. 
create or alter procedure CheckExamAccess_proc
    @examId int,
    @studentId int
as
begin
    declare @currentDateTime datetime = getdate();
    declare @startDateTime datetime;
    declare @endDateTime datetime;

    select @startDateTime = startTime, @endDateTime = endTime from Exam where e_id = @examId;

    if @currentDateTime >= @startDateTime and @currentDateTime <= @endDateTime
    begin
		
		if exists (select 1 from examStudent where examID = @examId and studentID = @studentId)
		begin

			select 'You can access the exam';          -- Allow student to access the exam

			select q.questionText 
			from question q
			inner join questionExam qe
			on q.ID = qe.questionID
			where qe.examID= @examId;

		end
		else
		begin
			select 'You do not have access to this exam';       -- Deny access to the exam
		end
    end
    else
    begin
        select 'The exam is not available at this time';               -- Deny access to the exam
	end;
end;

--test:
exec CheckExamAccess_proc 5,1 ;
exec CheckExamAccess_proc 4,1 ;




--stores student answers for each question id.
create or alter procedure StoreStudentAnswer_proc
    @studentId int,
    @examId int,
    @questionId int,
    @answerText nvarchar(max)
as
begin
    if exists (select 1 from examStudent where examID = @examId and studentID = @studentId)
    begin
        if exists (select 1 from questionExam where examID = @examId and questionID = @questionId)
        begin
            -- store the student's answer
            insert into [dbo].[examStudent] (studentID, examID, question_id, answer)
            values (@studentId, @examId, @questionId, @answerText);

            select 'Your answer has been stored' as message;
        end
        else 
        begin
            select 'The question does not belong to this exam' as message;
        end
    end
    else
    begin
        select 'You do not have access to this exam' as message;
    end
end;

--test:
exec StoreStudentAnswer_proc 1,5,1,'True';
exec StoreStudentAnswer_proc 1,5,4,'Mars';
exec StoreStudentAnswer_proc 1,5,13,'Structured query language';
exec StoreStudentAnswer_proc 1,5,25,'True';    --False
exec StoreStudentAnswer_proc 1,5,26,'True';
exec StoreStudentAnswer_proc 1,5,28,'Ottawa';
exec StoreStudentAnswer_proc 1,5,31,'Seoul';
exec StoreStudentAnswer_proc 1,5,32,'The cheetah';
exec StoreStudentAnswer_proc 1,5,38,'False';
exec StoreStudentAnswer_proc 1,5,39,'Marie Curie';
exec StoreStudentAnswer_proc 1,5,40,'Oxygen';
exec StoreStudentAnswer_proc 1,5,41,'cairo';      --London


--calculates if the answer if right.
create or alter procedure check_answer__calc_score_proc 
    @exam_id int, 
    @student_id int
as
begin
    update es
    set es.score = case 
                        when es.answer = q.questionanswer then qe.degree 
                        else 0 
                    end
    from 
        examstudent es
    join 
        question q on es.question_id = q.id
    join 
        questionexam qe on qe.questionid = q.id
    where 
        es.examid = @exam_id and es.studentid = @student_id;
end;


exec check_answer__calc_score_proc 5, 1;


--calculate final result for the student in this course.
create or alter proc calc_final_result_for_student 
	@exam_id int,
	@student_id int
as
begin
	select s.name , es.examID, sum(es.score) AS final_score
	from examStudent es
	join Student s
	on s.id = es.studentID
	where es.examID = @exam_id and es.studentID = @student_id
	group by s.name , es.examID
end;


--test:
exec calc_final_result_for_student 5, 1;
exec calc_final_result_for_student 3, 2;




--view to show the instructor and the courses 

	 CREATE VIEW ins_course AS
SELECT 
    i.[name] AS instructor_name,
    i.[ins_id] AS instructor_id,
    c.[name] AS course_name
FROM 
    [dbo].[Instructor] i
JOIN 
    [dbo].[Course] c ON i.ins_id = c.instructorID;

select * from [dbo].[ins_course];
	 --------------------------------------------
	
	 ---------------------
	 --view to show the intake name and thier tracks
	
	CREATE VIEW intake_track AS
SELECT 
    I.[name] AS intake_name,
    t.[name] AS track_name
FROM 
    [dbo].[Intake] I
JOIN 
    [dbo].[Track] t ON I.id = t.[intake_id];

select * from 	[dbo].[intake_track];

	-----------------------------
	--view to show the students and their courses  

	 CREATE VIEW studentcourses AS
SELECT 
    s.[name] AS student_name,
    s.[id] AS student_id,
    c.[name] AS course_name
FROM 
    [dbo].[Student] s
JOIN 
    [dbo].[studentCourse] cs ON s.id = cs.studentID
JOIN 
    [dbo].[Course] c ON cs.courseID = c.c_id;


--test

select * from [dbo].[studentcourses];

------------------------------------------------------------------------------------------------

--training manager permissions:

grant exec on [dbo].[add_branch_proc] to [training manager];
grant exec on [dbo].[add_intake_proc] to [training manager];
grant exec on [dbo].[add_student_proc] to [training manager];
grant exec on [dbo].[Add_Student_To_Course_PROC] to [training manager];
grant exec on [dbo].[add_track_branch_proc] to [training manager];
grant exec on [dbo].[add_track_proc] to [training manager];
grant exec on [dbo].[edit_branch_name_proc] to [training manager];
grant exec on [dbo].[edit_track_name_proc] to [training manager];
grant exec on [dbo].[insert_all_student] to [training manager];



--instructor permissions:
grant exec on [dbo].[select_students_for_exam_proc] to [instructor];
grant exec on [dbo].[chose_Q_random_proc] to [instructor];
grant exec on [dbo].[make_exam_manual_proc] to [instructor];
grant exec on [dbo].[make_exam_proc] to [instructor];
grant exec on [dbo].[disp_questions_proc] to [instructor];




--student permissions:

grant exec on [dbo].[CheckExamAccess_proc] to [student];



