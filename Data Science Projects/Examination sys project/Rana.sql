
-- creating tables

create table question
(
	ID int primary key,
	type nchar(3), -- TF, MCQ, TXT
	questionText nvarchar(max),
	questionAnswer nvarchar(max)
);


create table questionExam
(
	questionID int,
	examID int,
	degree int
);

create table examStudent
(
	examID int,
	studentID int
);

create table studentAnswer
(
	questionID int,
	studentID int,
	examID int,
	answer nvarchar(max),
	score int
)


-----------------------------------------------
use [Examination]
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

use [Examination]
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

use [Examination]

-- add few branches

exec dbo.disp_branch_proc

exec dbo.add_branch_proc 'Smart Village'
exec dbo.add_branch_proc 'Assuit'
exec dbo.add_branch_proc 'Ismailia'
exec dbo.add_branch_proc 'Menofia'
exec dbo.add_branch_proc 'Alexandria'
exec dbo.add_branch_proc 'Aswan'
exec dbo.add_branch_proc 'Mansoura'
exec dbo.add_branch_proc 'Minya'

exec dbo.disp_branch_proc

-- add intake

exec dbo.disp_intake_proc

exec dbo.add_intake_proc 'one', 'Smart Village'
exec dbo.add_intake_proc 'two', 'Smart Village'
exec dbo.add_intake_proc 'three', 'Smart Village'
exec dbo.add_intake_proc 'four', 'Smart Village'
exec dbo.add_intake_proc 'five', 'Smart Village'

exec dbo.add_intake_proc '2020Q1', 'Assuit'
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

exec dbo.disp_intake_proc 

-- add track

exec add_track_branch_proc 'OS','one', 'Smart Village'
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


-- display all

exec dbo.disp_all_proc


use [Examination]
go

exec disp_student_proc

exec add_student_proc 'Rana', 'Minia Uni', '01097363250', '2023', 'BI', '2023Q2', 'Minya'
exec add_student_proc 'Mohamed', 'Assuit Uni', '01012345678', '2020', 'python', '2023Q2', 'Minya'
exec add_student_proc 'Ahmed', 'Cairo Uni', '01112345678', '2021', '.net', '2023Q2', 'Minya'
exec add_student_proc 'Moaz', 'Alexandria Uni', '01212345678', '2022', 'web', '2023Q2', 'Minya'

exec disp_student_proc

exec add_student_proc 'test', 'Alexandria Uni', '01212345678', '2022', 'web', '2023Q2', 'Minya'
exec disp_student_proc
exec delete_student_proc 'test', 'web', '2023Q2', 'Minya'
exec disp_student_proc


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
select * from courses_for_python

go
create or alter view courses_for_dot_net
as 
select 
c_id, c.name, t.id
from Course c, Track t 
where c_id in (1,7) and t.id = 46;

go
select * from courses_for_dot_net

go
create or alter view courses_for_web
as 
select 
c_id, t.id
from Course c, Track t  
where c_id in (7) and t.id = 47;

go
select * from courses_for_web

--------------------------------------------------------------------------------------

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

select * from view_student_courses_func()

--------------------------------------------------------------------------------------------------
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

exec make_exam_proc @ins_id = 8, @c_id = 1, @type = 0,
@year = '2023', @allownace = 'half hour extra', 
@startTime = '2023-12-1 12:00', @endTime = '2023-12-1 14:00'

-- select * from Exam

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

exec disp_questions_proc
declare @eid int = (select IDENT_CURRENT('Exam'))
exec make_exam_manual_proc @exam_id = @eid, @c_id = 2,
@Question_id_string = '1,2,3,4,5,6,7', @question_degree_string = '1,2,3,4,5,6,7'

select * from questionExam

------------------------------------------------------------------------------------------------------------
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
------------------------------------------------------------------------------------------------------------------------------------
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
---------------------------------------------------------------------------------------------------------------
