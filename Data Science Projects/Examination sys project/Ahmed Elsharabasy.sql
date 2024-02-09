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
	intake_id int
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


--add relation 
alter table examStudent 
add constraint examStudent_question
foreign key (question_id)
references question(ID);

/*  System should store courses information (Course name, description, Max degree, Min Degree),
instructors’ information, and students’ information, each instructor can teach one or more course,
and each course may be teacher by one instructor in each class (Instructor may be changed for
other class in other year).  */

use Examination;


--select * from [dbo].[Course]

--select * from [dbo].[Instructor]

--select * from [dbo].[Student]

--select * from [dbo].[question]


--- instructor  (creating tasks)

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
    IF EXISTS (SELECT @id FROM [Instructor] WHERE ins_id= @id and [name] = @old_name)
    BEGIN
        UPDATE [Instructor]
        SET [Name] = @new_name
           WHERE ins_id = @ID and [name] = @old_name;

        PRINT 'Instructor: ' + @new_name + ' is updated';
    END
    ELSE
        PRINT 'Instructor does not exist';
end;

go

create proc disp_instructor_proc
as
begin
	select * from [dbo].[Instructor];
end;



--- instructor  (add data)    &&& test


exec [dbo].[add_instructor_proc] 'ahmed'
exec [dbo].[add_instructor_proc] 'ali'
exec [dbo].[add_instructor_proc] 'khaled'
exec [dbo].[add_instructor_proc] 'abdo'
exec [dbo].[add_instructor_proc] 'saeed'
exec [dbo].[add_instructor_proc] 'mostafa'
exec [dbo].[add_instructor_proc] 'moaaz'
exec [dbo].[add_instructor_proc] 'taha'


exec [dbo].[disp_instructor_proc]

exec [dbo].[edit_instructor_proc] 'taha' , 'rami', 8
exec [dbo].[disp_instructor_proc]

exec [dbo].[delete_instructor_proc] 16, 'taha'

exec [dbo].[add_instructor_proc] 'Rana'
exec [dbo].[delete_instructor_proc] 9, 'Rana'
exec [dbo].[disp_instructor_proc]


-------------------------------------------------------------------------------------------

--- Course  (creating tasks)
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



go
create proc disp_course_proc
as
begin
	select * from [dbo].[Course];
end;


--- Course  (add data)    &&& test

exec add_course_proc 'SQL', 'Basic database concepts', 0, 100, 8;
exec add_course_proc 'Python', 'Fundamentals of programming', 0, 100, 1;
exec add_course_proc 'Power BI', 'Building interactive dashboards', 0, 100, 2;
exec add_course_proc 'Data Science Fundamentals', 'Introduction to data analysis', 0, 100, 3;
exec add_course_proc 'Network Security', 'Securing computer networks', 0, 100, 8;
exec add_course_proc 'Data Mining', 'Overview of statistics and AI techniques', 0, 100, 1;
exec add_course_proc 'Web design basics', 'Overview of HTML and CSS', 0, 100,4;


exec delete_course_proc 5;

exec edit_course_proc 'SQL' , 'fundamentals of database concepts', 0, 100, 8


exec disp_course_proc;


select * from [dbo].[get_course_info_by_name_fun]('SQL');

select * from dbo.get_instructor_courses_name_id_func(1, 'ahmed')

------------------------------------------------------------------------------------------------------------------------------

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
exec select_students_for_exam_proc 2, 3
exec select_students_for_exam_proc 4, 3
exec select_students_for_exam_proc 6, 3
exec [dbo].[select_students_for_exam_proc] 3, 3

-------------------------------------------------------------------------------------------------

--Students can see the exam and do it only on the specified time.

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


exec CheckExamAccess_proc 5,1 ;

exec CheckExamAccess_proc 4,1 ;

--------------------------------------------------------------------------------------------------

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

----------------------------------------------------------------------------------------------------


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

---------------------------------------------------------------------------------------------

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
-----------------------------------------------------------------------------------------------

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

